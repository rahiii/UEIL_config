#!/usr/bin/env python3
"""
Automatically onboard a model into the framework.

Usage:
    python scripts/add_model.py --name <model>
    python scripts/add_model.py --name <model> --git https://github.com/...
    python scripts/add_model.py --name <model> --force   # overwrite existing config

Scans models/<name>/ with Python's AST module to auto-detect EVERYTHING:
  - Best nn.Module class (matches model name)
  - Constructor pattern (kwargs vs argparse.Namespace) + default values
  - Forward signature → input mode, output mode, extra kwargs

Generates a complete configs/<name>.yaml with zero manual editing.
"""

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Sentinels ────────────────────────────────────────────────────────────

# Marker: AST node exists but can't be resolved to a Python literal
# (e.g.  activations=nn.ReLU  — has a default, but not serializable to YAML)
_UNRESOLVED = object()

# Marker: parameter has NO default at all (truly required)
_REQUIRED = object()


# ── Constants ────────────────────────────────────────────────────────────

# Names that indicate the constructor takes a single namespace/config object
NAMESPACE_PARAM_NAMES = {
    "args", "opt", "options", "config", "cfg", "hparams", "opts", "params",
}

# Forward kwargs the framework handles internally — exclude from config
EXCLUDE_FORWARD_KWARGS = {
    "flow_init", "test_mode", "eval_mode", "inference_mode",
    "is_test", "training", "is_training", "train_mode",
}

# Constructor params the framework already handles — exclude from config
EXCLUDE_CONSTRUCTOR_PARAMS = {"device"}

# Sensible defaults for common constructor param names (when code has no default)
PARAM_GUESSES = {
    "inshape": [256, 256],
    "input_shape": [256, 256],
    "img_size": [256, 256],
    "image_size": [256, 256],
    "vol_size": [256, 256],
    "spatial_dims": 2,
    "ndims": 2,
    "ndim": 2,
    "n_dims": 2,
    "in_channels": 1,
    "source_channels": 1,
    "target_channels": 1,
    "channels": 1,
    "input_channels": 1,
    "n_channels": 1,
    "num_channels": 1,
}

# Params that describe the DATA, not the model — repo usage values are
# unreliable for these because the repo's data may differ from the user's
# data.  For these params we always use PARAM_GUESSES (derived from
# data.resize in the config) and NEVER copy from the repo's scripts.
DATA_DEPENDENT_PARAMS = {
    "ndim", "ndims", "n_dims", "spatial_dims",
    "inshape", "input_shape", "img_size", "image_size", "vol_size",
}


# ── AST Helpers ──────────────────────────────────────────────────────────


def _ast_literal(node: ast.AST):
    """Try to extract a Python literal from an AST node.

    Returns the literal value (including None for Python None),
    or _UNRESOLVED if the node can't be converted to a YAML-safe literal.
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass
    if isinstance(node, ast.Constant):
        return node.value          # handles None, int, float, str, bool, ...
    if isinstance(node, ast.Name):
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        if node.id == "None":
            return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _ast_literal(node.operand)
        if inner is not _UNRESOLVED:
            return -inner
    return _UNRESOLVED


def _yaml_val(v) -> str:
    """Format a Python value as a YAML scalar."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v == 0.0:
            return "0.0"
        if 0 < abs(v) < 0.01 or abs(v) >= 10000:
            return f"{v:.1e}"
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_yaml_val(x) for x in v) + "]"
    if v is None:
        return "null"
    return str(v)


# ── AST-Based Model Scanner ─────────────────────────────────────────────


def _scan_py_file(py_path: Path, model_dir: Path) -> List[Dict]:
    """Parse a .py file and return info about every nn.Module subclass."""
    try:
        source = py_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check inheritance for nn.Module / Module / Network / Net
        is_module = False
        is_autograd_func = False
        for base in node.bases:
            base_name = ""
            if isinstance(base, ast.Attribute):
                base_name = base.attr
            elif isinstance(base, ast.Name):
                base_name = base.id
            if "Module" in base_name or "Network" in base_name or "Net" in base_name:
                is_module = True
                break
            # torch.autograd.Function has forward() too but is NOT a model
            if base_name == "Function":
                is_autograd_func = True

        # Skip autograd Functions — they have forward() but aren't models
        if is_autograd_func:
            continue

        # Duck-typing: has BOTH __init__ and forward() methods
        if not is_module:
            has_init = False
            has_forward = False
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "__init__":
                        has_init = True
                    if item.name == "forward":
                        has_forward = True
            if has_init and has_forward:
                is_module = True

        if not is_module:
            continue

        info = _analyze_class(node, py_path, model_dir)
        if info:
            results.append(info)

    return results


def _analyze_class(cls_node: ast.ClassDef, py_path: Path,
                   model_dir: Path) -> Optional[Dict]:
    """Full analysis: constructor, forward signature, return type."""
    rel = py_path.relative_to(model_dir)
    mod_path = str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")
    entry_point = f"{mod_path}.{cls_node.name}"

    info: Dict[str, Any] = {
        "class_name": cls_node.name,
        "entry_point": entry_point,
        "file": str(py_path.relative_to(model_dir)),
        "args_mode": "kwargs",
        "init_params": [],
        "namespace_attrs": [],
        "namespace_defaults": {},
        "forward_params": [],
        "forward_extra_kwargs": {},
        "input_mode": "pair_rgb",
        "output_mode": "direct",
        "has_forward": False,
    }

    init_func = None
    for item in cls_node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name == "__init__":
            init_func = item
            _analyze_init(item, info)
        elif item.name == "forward":
            info["has_forward"] = True
            _analyze_forward(item, info)

    # For namespace mode: scan the ENTIRE class for self.args.xxx accesses
    # (catches attrs used in forward/other methods, not just __init__)
    if info["args_mode"] == "namespace":
        _scan_class_namespace_attrs(cls_node, info)
        if init_func:
            _extract_namespace_defaults(init_func, info)

    return info


def _analyze_init(func: ast.FunctionDef, info: Dict):
    """Detect kwargs vs namespace pattern and extract constructor params."""
    params = [a.arg for a in func.args.args if a.arg != "self"]
    info["init_params"] = params

    # Namespace pattern: single param named args/opt/config/etc.
    if len(params) == 1 and params[0].lower() in NAMESPACE_PARAM_NAMES:
        info["args_mode"] = "namespace"
        param_name = params[0]

        # Find all args.xxx attribute accesses in __init__
        attrs = set()
        for subnode in ast.walk(func):
            if (isinstance(subnode, ast.Attribute)
                    and isinstance(subnode.value, ast.Name)
                    and subnode.value.id == param_name):
                attrs.add(subnode.attr)
        info["namespace_attrs"] = sorted(attrs)

    elif len(params) >= 1:
        info["args_mode"] = "kwargs"
        defaults = func.args.defaults
        n_params = len(params)
        n_defaults = len(defaults)
        kw_params = []
        for i, p in enumerate(params):
            default_idx = i - (n_params - n_defaults)
            if default_idx >= 0:
                val = _ast_literal(defaults[default_idx])
                # val is _UNRESOLVED → has default but not serializable (e.g. nn.ReLU)
                # val is None        → default is Python None
                # val is <literal>   → concrete default
                kw_params.append((p, val))
            else:
                # No default at all → truly required
                kw_params.append((p, _REQUIRED))
        info["init_params_with_defaults"] = kw_params


def _scan_class_namespace_attrs(cls_node: ast.ClassDef, info: Dict):
    """Scan entire class for self.args.xxx patterns (beyond __init__)."""
    all_attrs = set(info["namespace_attrs"])
    param_name = info["init_params"][0] if info["init_params"] else "args"

    for node in ast.walk(cls_node):
        if not isinstance(node, ast.Attribute):
            continue
        # Match: self.args.xxx
        if (isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
                and node.value.attr == param_name):
            all_attrs.add(node.attr)

    info["namespace_attrs"] = sorted(all_attrs)


def _extract_namespace_defaults(init_func: ast.FunctionDef, info: Dict):
    """Extract default values for namespace attrs from assignments in __init__.

    Handles patterns like:
        args.small = False
        args.corr_levels = 4
        self.args.dropout = 0
        if 'dropout' not in self.args:
            self.args.dropout = 0
    """
    param_name = info["init_params"][0]
    assignments = []  # (attr_name, value, lineno)

    for node in ast.walk(init_func):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            attr_name = _get_namespace_target(target, param_name)
            if attr_name:
                val = _ast_literal(node.value)
                if val is not _UNRESOLVED:
                    assignments.append((attr_name, val,
                                        getattr(node, "lineno", 0)))

    # Sort by line number → last assignment wins (handles if/else branches;
    # the else branch is the "default" path and comes last in source)
    assignments.sort(key=lambda x: x[2])
    defaults = {}
    for attr, val, _ in assignments:
        defaults[attr] = val

    # Attrs that are only READ (never assigned) → assume boolean flag, False
    for attr in info["namespace_attrs"]:
        if attr not in defaults:
            defaults[attr] = False

    info["namespace_defaults"] = defaults


def _get_namespace_target(target, param_name: str) -> Optional[str]:
    """Return attr name if target is  args.xxx  or  self.args.xxx ."""
    if not isinstance(target, ast.Attribute):
        return None
    # args.xxx
    if isinstance(target.value, ast.Name) and target.value.id == param_name:
        return target.attr
    # self.args.xxx
    if (isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "self"
            and target.value.attr == param_name):
        return target.attr
    return None


def _detect_rgb_split(func: ast.FunctionDef) -> bool:
    """Check if a single-arg forward splits the input at channel 3 (RGB).

    Looks for patterns like x[:, :3], x[:, 3:], x[:, :3, :, :] etc.
    Returns True if the model expects 6-channel (3+3 RGB) stacked input.
    """
    for node in ast.walk(func):
        if not isinstance(node, ast.Subscript):
            continue
        # Look for slicing patterns with constant 3
        sl = node.slice
        if isinstance(sl, ast.Tuple):
            for elt in sl.elts:
                if isinstance(elt, ast.Slice):
                    for bound in (elt.lower, elt.upper):
                        if isinstance(bound, ast.Constant) and bound.value == 3:
                            return True
        elif isinstance(sl, ast.Slice):
            for bound in (sl.lower, sl.upper):
                if isinstance(bound, ast.Constant) and bound.value == 3:
                    return True
    return False


def _analyze_forward(func: ast.FunctionDef, info: Dict):
    """Detect input mode, output mode, and extra call kwargs."""
    params = [a.arg for a in func.args.args if a.arg != "self"]
    info["forward_params"] = params

    # ── Input mode ──────────────────────────────────────
    img_params = params[:2] if len(params) >= 2 else params[:1]
    pair_gray = {"source", "target", "moving", "fixed", "src", "tgt"}
    pair_rgb = {"image1", "image2", "img1", "img2", "im1", "im2",
                "frame1", "frame2", "x1", "x2"}

    if len(img_params) >= 2:
        names = {p.lower() for p in img_params}
        if names & pair_gray:
            info["input_mode"] = "pair_gray"
        elif names & pair_rgb:
            info["input_mode"] = "pair_rgb"
        else:
            info["input_mode"] = "pair_rgb"  # default for two-arg forward
    elif len(img_params) == 1:
        if img_params[0].lower() in ("x", "input", "inp", "img", "images"):
            # Detect if the model slices x[:, :3] / x[:, 3:] → needs 6ch RGB
            # or x[:, :1] / x[:, 1:] → 2ch grayscale
            needs_rgb = _detect_rgb_split(func)
            info["input_mode"] = "stack6ch" if needs_rgb else "stack2ch"

    # ── Extra call kwargs (beyond image args) ───────────
    n_img = 2 if info["input_mode"].startswith("pair") else 1
    extra_params = params[n_img:]

    n_params = len(params)
    defaults = func.args.defaults
    n_defaults = len(defaults)
    extra_kw = {}

    for i, p in enumerate(extra_params):
        abs_idx = n_img + i
        def_idx = abs_idx - (n_params - n_defaults)
        if def_idx >= 0:
            val = _ast_literal(defaults[def_idx])
            if val is not _UNRESOLVED and val is not None:
                extra_kw[p] = val

    # Keyword-only args
    for kw_arg, kw_def in zip(func.args.kwonlyargs, func.args.kw_defaults):
        if kw_def is not None:
            val = _ast_literal(kw_def)
            if val is not _UNRESOLVED and val is not None:
                extra_kw[kw_arg.arg] = val

    info["forward_extra_kwargs"] = extra_kw

    # ── Output mode (from return statements) ────────────
    returns = []
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            returns.append(node)

    if returns:
        # Sort by line number, use LAST return (the default/fallthrough path)
        returns.sort(key=lambda n: getattr(n, "lineno", 0))
        last_ret = returns[-1].value

        if isinstance(last_ret, ast.Tuple):
            n = len(last_ret.elts)
            if n == 2:
                idx = _guess_flow_index(last_ret)
                info["output_mode"] = f"tuple_index:{idx}"
            else:
                info["output_mode"] = "tuple_first"

        elif isinstance(last_ret, ast.List):
            info["output_mode"] = "list_last"

        elif isinstance(last_ret, ast.Name):
            # Variable that might be a list (check for .append() pattern)
            for node in ast.walk(func):
                if isinstance(node, ast.Attribute) and node.attr == "append":
                    info["output_mode"] = "list_last"
                    break

        elif isinstance(last_ret, ast.Subscript):
            # e.g. return predictions[-1]
            info["output_mode"] = "direct"

        elif isinstance(last_ret, ast.Dict):
            for key in last_ret.keys:
                if isinstance(key, ast.Constant):
                    k = str(key.value).lower()
                    if "flow" in k or "disp" in k or "field" in k:
                        info["output_mode"] = f"dict_key:{key.value}"
                        break


def _guess_flow_index(tuple_node: ast.Tuple) -> int:
    """Guess which tuple element is the flow/displacement."""
    flow_names = {"flow", "displacement", "disp", "field", "warp_field",
                  "flow_up", "flow_pr", "flow_pred", "deformation", "ddf"}
    for i, elt in enumerate(tuple_node.elts):
        name = ""
        if isinstance(elt, ast.Name):
            name = elt.id.lower()
        elif isinstance(elt, ast.Attribute):
            name = elt.attr.lower()
        if any(fn in name for fn in flow_names):
            return i
    return len(tuple_node.elts) - 1


# ── Usage Scanner (how does the model's OWN code instantiate itself?) ────


def _scan_usage_kwargs(model_dir: Path, class_name: str) -> Dict[str, Any]:
    """Scan the model repo's own scripts/tests for constructor calls.

    Looks for e.g.  VxmPairwise(ndim=3, source_channels=1, ...)
    and extracts the literal kwarg values — so we use the SAME values
    the authors use instead of guessing.
    """
    found: Dict[str, Any] = {}

    for py_file in sorted(model_dir.rglob("*.py")):
        rel = str(py_file.relative_to(model_dir))
        if "__pycache__" in rel or rel == "setup.py":
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            # Match calls like  ClassName(...)  or  module.ClassName(...)
            callee = ""
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr

            if callee != class_name:
                continue

            # Extract keyword arguments with literal values
            for kw in node.keywords:
                if kw.arg is None:
                    continue  # **kwargs splat
                val = _ast_literal(kw.value)
                if val is not _UNRESOLVED and kw.arg not in found:
                    found[kw.arg] = val

            # Extract positional arguments (match against known param order)
            # — less reliable, only use for the first few common ones
            # (most repos use keyword style anyway)

    return found


def _scan_argparse_defaults(model_dir: Path) -> Dict[str, Any]:
    """Scan model repo for argparse add_argument calls to find defaults.

    Handles patterns like:
        parser.add_argument('--small', action='store_true')  → small: False
        parser.add_argument('--dropout', type=float, default=0.0)  → dropout: 0.0
        parser.add_argument('--iters', type=int, default=12)  → iters: 12
    """
    found: Dict[str, Any] = {}

    for py_file in sorted(model_dir.rglob("*.py")):
        rel = str(py_file.relative_to(model_dir))
        if "__pycache__" in rel or rel == "setup.py":
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Match: parser.add_argument(...)  or  *.add_argument(...)
            if not (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue

            # Get the argument name from the first positional arg
            if not node.args:
                continue
            first_arg = node.args[0]
            if not isinstance(first_arg, ast.Constant):
                continue
            arg_name = str(first_arg.value)
            if not arg_name.startswith("--"):
                continue
            arg_name = arg_name.lstrip("-").replace("-", "_")

            # Check for action='store_true' → default is False
            action = None
            default = _UNRESOLVED
            for kw in node.keywords:
                if kw.arg == "action":
                    val = _ast_literal(kw.value)
                    if val is not _UNRESOLVED:
                        action = val
                elif kw.arg == "default":
                    val = _ast_literal(kw.value)
                    if val is not _UNRESOLVED:
                        default = val

            if action == "store_true":
                found[arg_name] = False
            elif action == "store_false":
                found[arg_name] = True
            elif default is not _UNRESOLVED:
                found[arg_name] = default

    return found


# ── Nested-repo helpers ─────────────────────────────────────────────────


def _compute_sys_path(model_file_rel: str, model_dir: Path) -> str:
    """Determine if a subdirectory needs to be added to sys.path.

    Many research repos have internal imports like ``import models.configs_X``
    that assume the project sub-root is on sys.path.  When the selected model
    class lives in a nested subdirectory we detect those imports and return the
    subdirectory that should be prepended to sys.path.

    Returns a relative path from *model_dir* (e.g. ``"RaFD/TransMorph2D"``),
    or ``""`` if no adjustment is needed.
    """
    abs_path = model_dir / model_file_rel
    if not abs_path.exists():
        return ""
    try:
        source = abs_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return ""

    # Collect top-level import names (absolute imports only)
    import_tops: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_tops.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            import_tops.add(node.module.split(".")[0])

    # Walk the file's ancestor directories inside model_dir.
    # If any directory name matches a top-level import, its PARENT is the
    # sys_path we need.
    parts = Path(model_file_rel).parts  # e.g. ('RaFD','TransMorph2D','models','TransMorph.py')
    for i in range(len(parts) - 1):     # skip the filename itself
        if parts[i] in import_tops:
            parent_parts = parts[:i]
            return str(Path(*parent_parts)) if parent_parts else ""

    return ""


def _scan_config_factory_defaults(model_file_rel: str,
                                  model_dir: Path) -> Dict[str, Any]:
    """Scan config-factory files near the model source for default values.

    Handles patterns like ``ml_collections.ConfigDict()`` with assignments
    such as ``config.embed_dim = 96``.  Prefers functions whose name
    contains "2d" or "2D" (matching the user's 2-D data).

    Returns a dict of {attr_name: value}.
    """
    abs_path = model_dir / model_file_rel
    parent_dir = abs_path.parent

    # Locate candidate config files in the same directory
    config_files: List[Path] = []
    for pattern in ("configs_*", "config_*", "cfg_*"):
        config_files.extend(parent_dir.glob(pattern + ".py"))
    config_files = sorted(set(config_files))

    if not config_files:
        return {}, False

    best_defaults: Dict[str, Any] = {}
    best_is_2d = False
    uses_ml_collections = False

    for cf in config_files:
        try:
            source = cf.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        if "ml_collections" in source:
            uses_ml_collections = True

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            func_name = node.name.lower()
            is_2d = "2d" in func_name
            is_3d = "3d" in func_name

            # Skip pure-3D config functions
            if is_3d and not is_2d:
                continue

            # Extract  config.xxx = value  assignments
            func_defaults: Dict[str, Any] = {}
            for stmt in ast.walk(node):
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id in
                            ("config", "cfg", "opt", "args", "c")):
                        val = _ast_literal(stmt.value)
                        if val is not _UNRESOLVED:
                            func_defaults[target.attr] = val

            if not func_defaults:
                continue

            # Prefer 2-D configs over everything else
            if is_2d and not best_is_2d:
                best_defaults = func_defaults
                best_is_2d = True
            elif not best_defaults:
                best_defaults = func_defaults

    return best_defaults, uses_ml_collections


# ── Scanning & Ranking ──────────────────────────────────────────────────


def scan_model_dir(model_dir: Path) -> List[Dict]:
    """Scan all .py files for nn.Module subclasses."""
    # Directories that should never contain the main model class:
    #   build artifacts, external deps, non-PyTorch frameworks, Docker, etc.
    SKIP_DIR_PARTS = {
        "__pycache__", "build", "dist", "tmp", ".egg", "egg-info",
        "docker", "external_packages", "caffe", "tensorflow",
        "benchmark", "deprecated", "old", "archive",
    }

    results = []
    for py_file in sorted(model_dir.rglob("*.py")):
        rel = str(py_file.relative_to(model_dir))
        rel_lower = rel.lower()
        # Skip files in junk directories
        parts_lower = {p.lower() for p in Path(rel).parts[:-1]}  # dir parts only
        if parts_lower & SKIP_DIR_PARTS:
            continue
        # Skip test files and setup.py
        if rel == "setup.py":
            continue
        results.extend(_scan_py_file(py_file, model_dir))
    return results


def _rank_candidates(candidates: List[Dict], model_name: str = "") -> List[Dict]:
    """Sort: most likely main model class first."""

    # Directories that suggest "not the main model" even if they pass scanning
    PENALTY_DIR_PARTS = {
        "docker", "caffe", "tensorflow", "external", "third_party",
        "legacy", "deprecated", "old", "archive", "benchmark",
        "multi_frame", "fusion",
    }

    def score(c):
        s = 0
        if c["has_forward"]:
            s += 10
        name_lower = c["class_name"].lower()
        file_lower = c["file"].lower()
        file_parts = {p.lower() for p in Path(c["file"]).parts[:-1]}

        # Huge bonus for exact name match with model directory
        if model_name and name_lower == model_name.lower():
            s += 50
        elif model_name and (model_name.lower() in name_lower
                             or name_lower in model_name.lower()):
            s += 20
        # Prefer 2D models (user's data is always 2D grayscale)
        if "2d" in file_lower or "2d" in name_lower:
            s += 30
        # Penalise 3D models
        if "3d" in file_lower and "2d" not in file_lower:
            s -= 15
        # Penalise classes from non-main directories
        if file_parts & PENALTY_DIR_PARTS:
            s -= 20
        # Prefer shallow paths (main model is usually at the top level)
        depth = len(Path(c["file"]).parts) - 1
        s -= max(0, depth - 2) * 3  # penalty for deep nesting
        # Bonus for files in a "models/" subdirectory (common convention)
        if "models" in file_parts:
            s += 5
        # Bonus for files named after PyTorch convention
        if "pytorch" in file_lower or "torch" in file_lower:
            s += 5
        # Common model-class keywords
        for kw in ("net", "model", "dense", "flow"):
            if kw in name_lower:
                s += 5
        # Not in utils/helpers
        if "util" not in file_lower and "helper" not in file_lower:
            s += 3
        # More params → more configurable → likely main model
        s += min(len(c["init_params"]), 5)
        # Two-image forward → optical flow / registration model
        if len(c["forward_params"]) >= 2:
            s += 3
        return s
    return sorted(candidates, key=score, reverse=True)


def _auto_pick(candidates: List[Dict], model_name: str) -> Optional[Dict]:
    """Automatically pick the best model class — no user input."""
    if not candidates:
        return None
    with_forward = [c for c in candidates if c["has_forward"]]
    pool = with_forward if with_forward else candidates
    ranked = _rank_candidates(pool, model_name)
    return ranked[0]


# ── Config Generation ───────────────────────────────────────────────────


def _generate_config(name: str, info: Dict) -> str:
    """Generate a complete, ready-to-run YAML config."""
    entry_point = info["entry_point"]
    args_mode = info["args_mode"]
    input_mode = info["input_mode"]
    output_mode = info["output_mode"]

    # Build the YAML line-by-line for correct indentation
    L = []  # lines accumulator

    L.append(f"# Auto-generated by:  python scripts/add_model.py --name {name}")
    L.append(f"# Detected from:      {info['file']} -> {info['class_name']}")
    L.append("#")
    L.append(f"# To regenerate:      python scripts/add_model.py --name {name} --force")
    L.append("")
    L.append("# -- Model ----------------------------------------------------------------")
    L.append("#    framework/adapter.py imports this class and constructs it")
    L.append("model:")
    L.append(f"  name: {name}")
    L.append(f"  entry_point: {entry_point}")
    L.append(f"  args_mode: {args_mode}")
    sys_path = info.get("sys_path", "")
    if sys_path:
        L.append(f"  sys_path: {sys_path}")

    # Model args
    if args_mode == "namespace":
        defaults = info.get("namespace_defaults", {})
        attrs = info.get("namespace_attrs", [])
        if attrs:
            L.append("  args:")
            for attr in attrs:
                # Override data-dependent params with user's data settings
                if attr.lower() in DATA_DEPENDENT_PARAMS and attr.lower() in PARAM_GUESSES:
                    val = PARAM_GUESSES[attr.lower()]
                else:
                    val = defaults.get(attr, False)
                L.append(f"    {attr}: {_yaml_val(val)}")
        else:
            L.append("  args: {}")
    else:
        pairs = info.get("init_params_with_defaults", [])
        usage = info.get("usage_kwargs", {})
        kept = []
        for param, default in pairs:
            # Skip params the framework handles (e.g. device)
            if param.lower() in EXCLUDE_CONSTRUCTOR_PARAMS:
                continue
            # Skip non-literal defaults (e.g. nn.ReLU) — let Python use its own
            if default is _UNRESOLVED:
                continue
            # Skip None defaults — let Python default to None on its own
            if default is None:
                continue
            kept.append((param, default))
        if kept:
            L.append("  args:")
            for param, default in kept:
                if param.lower() in DATA_DEPENDENT_PARAMS:
                    # These depend on the USER's data, not the repo's data
                    if param.lower() in PARAM_GUESSES:
                        L.append(f"    {param}: {_yaml_val(PARAM_GUESSES[param.lower()])}")
                    elif default is not _REQUIRED:
                        L.append(f"    {param}: {_yaml_val(default)}")
                    else:
                        L.append(f"    {param}: null  # required - set this value")
                elif param in usage:
                    # Use the value from the model's OWN code
                    L.append(f"    {param}: {_yaml_val(usage[param])}")
                elif default is not _REQUIRED:
                    L.append(f"    {param}: {_yaml_val(default)}")
                elif param.lower() in PARAM_GUESSES:
                    L.append(f"    {param}: {_yaml_val(PARAM_GUESSES[param.lower()])}")
                else:
                    L.append(f"    {param}: null  # required - set this value")
        else:
            L.append("  args: {}")

    L.append("")
    L.append("# -- Forward ---------------------------------------------------------------")
    L.append("#    framework/adapter.py calls model(img1, img2, **call_kwargs)")
    L.append("forward:")
    L.append(f"  input: {input_mode}")

    # Call kwargs
    call_kwargs = {}
    for k, v in info.get("forward_extra_kwargs", {}).items():
        if k.lower() not in EXCLUDE_FORWARD_KWARGS and v is not None:
            call_kwargs[k] = v
    if call_kwargs:
        L.append("  call_kwargs:")
        for k, v in call_kwargs.items():
            L.append(f"    {k}: {_yaml_val(v)}")
    else:
        L.append("  call_kwargs: {}")

    L.append(f"  output: {output_mode}")
    L.append("  output_format: flow_2hw")

    L.append("")
    L.append("# -- Data ------------------------------------------------------------------")
    L.append("#    framework/data.py loads image pairs from this path")
    L.append("data:")
    L.append("  path: data")
    L.append('  pattern: "*.mat"')
    L.append("  key: img")
    L.append("  resize: [256, 256]")
    L.append("  normalization: none")
    L.append("  select:")
    L.append("    file_range: null")
    L.append("    pair_indices: null")
    L.append("    max_pairs: null")

    L.append("")
    L.append("# -- Training --------------------------------------------------------------")
    L.append("#    train.py + framework/loss.py")
    L.append("training:")
    L.append("  batch_size: 4")
    L.append("  lr: 1.0e-4")
    L.append("  num_steps: 200")
    L.append("  validation_split: 0.2")
    L.append("  val_freq: 25")
    L.append("  loss:")
    L.append("    type: photometric_smooth")
    L.append("    w_photo: 1.0")
    L.append("    w_smooth: 0.05")

    L.append("")
    L.append("# -- Runtime ---------------------------------------------------------------")
    L.append("runtime:")
    L.append("  device: auto")
    L.append("  num_workers: 0")
    L.append("  log_freq: 5")

    L.append("")
    L.append("# -- Output ----------------------------------------------------------------")
    L.append("output:")
    L.append(f"  run_dir: runs/{name}_train")
    L.append("  save_checkpoint_freq: 50")
    L.append("  save_vis_freq: 50")
    L.append("  inference:")
    L.append(f"    outdir: outputs/{name}_disp")
    L.append("    format: single_mat")
    L.append(f"    filename: {name}_disp_all.mat")
    L.append("    save_vis: true")
    L.append("    skip_existing: true")

    L.append("")
    L.append("# -- Environment (used by scripts/setup.sh) ---------------------------------")
    L.append("setup:")
    L.append('  python: "3.10"')
    L.append("  conda: [numpy, scipy, matplotlib, pyyaml]")
    extra_pip = info.get("_extra_pip", [])
    pip_list = ["torch", "torchvision"] + [p for p in extra_pip if p not in ("torch", "torchvision")]
    L.append("  pip: [" + ", ".join(pip_list) + "]")
    L.append("")

    return "\n".join(L)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Auto-generate a model config. "
                    "Scans source code, no manual input needed.",
    )
    ap.add_argument("--name", required=True,
                    help="Model name (must match models/<name>/ directory)")
    ap.add_argument("--git", default="",
                    help="Git URL to clone into models/<name>/ if missing")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing config file")
    cli = ap.parse_args()

    name = cli.name.lower().strip().replace(" ", "_")
    config_path = REPO_ROOT / "configs" / (name + ".yaml")
    model_dir = REPO_ROOT / "models" / name

    print()
    print("=" * 60)
    print("  Onboarding: " + name)
    print("=" * 60)
    print()

    # Guard: existing config
    if config_path.exists() and not cli.force:
        print("  configs/" + name + ".yaml already exists.")
        print("  Run with --force to overwrite.")
        sys.exit(1)

    # Clone if needed
    if not model_dir.exists():
        if cli.git:
            print("  Cloning into models/" + name + "/ ...")
            subprocess.run(
                ["git", "clone", cli.git, str(model_dir)], check=True,
            )
            print()
        else:
            print("  Error: models/" + name + "/ not found.")
            print("  Either place the code there or use:")
            print("    --git https://github.com/...")
            sys.exit(1)

    # Scan model source code
    print("  Scanning models/" + name + "/ ...")
    candidates = scan_model_dir(model_dir)

    if not candidates:
        print("  Error: no nn.Module classes found in models/" + name + "/.")
        sys.exit(1)

    detected = _auto_pick(candidates, name)
    if not detected:
        print("  Error: could not determine model class.")
        sys.exit(1)

    # Show what was detected
    print("  Auto-selected: " + detected["class_name"])
    print("    from:        " + detected["file"])
    print("    entry_point: " + detected["entry_point"])

    # ── Compute sys_path for repos with nested internal imports ────────
    sys_path = _compute_sys_path(detected["file"], model_dir)
    if sys_path:
        # Shorten the entry_point so it's relative to the sys_path dir
        prefix = sys_path.replace("/", ".").replace("\\", ".") + "."
        if detected["entry_point"].startswith(prefix):
            detected["entry_point"] = detected["entry_point"][len(prefix):]
        detected["sys_path"] = sys_path
        print("    sys_path:    " + sys_path)
        print("    adjusted ep: " + detected["entry_point"])
    print()

    # ── Scan the model's OWN scripts/tests for constructor usage ──────
    if detected["args_mode"] == "kwargs":
        usage_vals = _scan_usage_kwargs(model_dir, detected["class_name"])
        if usage_vals:
            print("  Found constructor usage in repo scripts:")
            for k, v in usage_vals.items():
                print(f"      {k}: {_yaml_val(v)}")
            print()
        detected["usage_kwargs"] = usage_vals
    elif detected["args_mode"] == "namespace":
        # For namespace models, also grab argparse defaults
        argparse_vals = _scan_argparse_defaults(model_dir)
        if argparse_vals:
            existing = detected.get("namespace_defaults", {})
            for k, v in argparse_vals.items():
                if k in detected.get("namespace_attrs", []):
                    existing[k] = v
            detected["namespace_defaults"] = existing

        # Scan config-factory files (e.g. ml_collections.ConfigDict)
        factory_result = _scan_config_factory_defaults(
            detected["file"], model_dir,
        )
        if isinstance(factory_result, tuple):
            factory_defaults, uses_ml_collections = factory_result
        else:
            factory_defaults, uses_ml_collections = factory_result, False
        if factory_defaults:
            print("  Found config factory defaults:")
            for k, v in sorted(factory_defaults.items()):
                print(f"      {k}: {_yaml_val(v)}")
            print()
            existing = detected.get("namespace_defaults", {})
            for k, v in factory_defaults.items():
                if k in detected.get("namespace_attrs", []):
                    existing[k] = v
            detected["namespace_defaults"] = existing
        detected["_extra_pip"] = []
        if uses_ml_collections:
            detected["_extra_pip"].append("ml-collections")

    if detected["args_mode"] == "namespace":
        defaults = detected.get("namespace_defaults", {})
        print("  Constructor: namespace")
        print("    attrs detected: " + ", ".join(detected["namespace_attrs"]))
        for attr in detected["namespace_attrs"]:
            val = defaults.get(attr, "?")
            vstr = _yaml_val(val) if val != "?" else "?"
            print("      " + attr + ": " + vstr)
    else:
        usage = detected.get("usage_kwargs", {})
        print("  Constructor: kwargs")
        for p, d in detected.get("init_params_with_defaults", []):
            if p.lower() in EXCLUDE_CONSTRUCTOR_PARAMS:
                print("      " + p + ": (skipped, framework handles)")
            elif d is _UNRESOLVED:
                print("      " + p + ": (skipped, code has non-literal default)")
            elif d is None:
                print("      " + p + ": (skipped, defaults to None)")
            elif p.lower() in DATA_DEPENDENT_PARAMS:
                if p.lower() in PARAM_GUESSES:
                    print("      " + p + ": " + _yaml_val(PARAM_GUESSES[p.lower()]) + " (from your data)")
                else:
                    print("      " + p + ": ? (data-dependent)")
            elif d is _REQUIRED:
                if p in usage:
                    print("      " + p + ": " + _yaml_val(usage[p]) + " (from repo)")
                elif p.lower() in PARAM_GUESSES:
                    print("      " + p + ": " + _yaml_val(PARAM_GUESSES[p.lower()]) + " (guessed)")
                else:
                    print("      " + p + ": ? (required)")
            else:
                # Has a literal default, but check if repo uses a different value
                if p in usage and usage[p] != d:
                    print("      " + p + ": " + _yaml_val(usage[p]) + " (from repo, default was " + _yaml_val(d) + ")")
                else:
                    print("      " + p + ": " + _yaml_val(d))
    print()

    fwd = ", ".join(detected["forward_params"]) if detected["forward_params"] else "?"
    print("  Forward: (" + fwd + ")")
    print("    input:  " + detected["input_mode"])
    print("    output: " + detected["output_mode"])

    call_kw = {
        k: v
        for k, v in detected.get("forward_extra_kwargs", {}).items()
        if k.lower() not in EXCLUDE_FORWARD_KWARGS and v is not None
    }
    if call_kw:
        kw_str = ", ".join(k + "=" + _yaml_val(v) for k, v in call_kw.items())
        print("    kwargs: " + kw_str)
    print()

    # Generate config
    config_text = _generate_config(name, detected)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_text)

    # Done
    print("=" * 60)
    print("  Done: configs/" + name + ".yaml")
    print()
    print("  Next:")
    print("    bash scripts/setup.sh configs/" + name + ".yaml")
    print("    conda activate " + name)
    print("    python train.py --config configs/" + name + ".yaml")
    print("    python inference.py --config configs/" + name + ".yaml")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
