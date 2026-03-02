"""
Onboarding script — scans a model's code, generates a working adapter.py
and config YAML, then optionally sets up the conda environment.

Usage:
    python create_adapter.py --name transmorph
    python create_adapter.py --name transmorph --setup   # also run setup_model.ps1
"""

import os
import sys
import ast
import argparse
import subprocess
import textwrap

ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask(prompt, default=None):
    suffix = " [%s]: " % default if default else ": "
    ans = input(prompt + suffix).strip()
    return ans if ans else default


def ask_choice(prompt, options):
    """Present numbered choices and return the selected item."""
    for i, opt in enumerate(options):
        print("  %d) %s" % (i + 1, opt))
    while True:
        raw = input(prompt + " [1]: ").strip()
        if not raw:
            return options[0]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


# ---------------------------------------------------------------------------
# AST-based code scanner
# ---------------------------------------------------------------------------

class ModelInfo:
    """Information extracted from a single .py file."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.classes = []       # list of dicts
        self.configs_dict = []  # list of (var_name, [key_strings])
        self.in_chans = None

    def __repr__(self):
        return "ModelInfo(%s, classes=%s)" % (self.filepath, [c['name'] for c in self.classes])


def _get_dotted_name(node):
    """Resolve ast node to a dotted name string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _get_dotted_name(node.value)
        if parent:
            return parent + '.' + node.attr
        return node.attr
    return ''


def _analyze_method(class_node, method_name):
    """Find a method in a class and return (required_param_count, param_names, node)."""
    for item in ast.iter_child_nodes(class_node):
        if isinstance(item, ast.FunctionDef) and item.name == method_name:
            args = item.args
            all_args = args.args
            # Remove 'self'
            if all_args and all_args[0].arg == 'self':
                all_args = all_args[1:]
            param_names = [a.arg for a in all_args]
            # Required params = total - defaults
            n_defaults = len(args.defaults)
            n_required = len(all_args) - n_defaults
            return n_required, param_names, item
    return 0, [], None


def _detect_return_pattern(forward_node):
    """Analyze forward() return to guess output format."""
    for node in ast.walk(forward_node):
        if isinstance(node, ast.Return) and node.value is not None:
            val = node.value
            if isinstance(val, ast.Tuple):
                return 'tuple', len(val.elts)
            if isinstance(val, ast.List):
                return 'list', len(val.elts)
            return 'single', 1
    return 'unknown', 0


def _scan_file(filepath):
    """Parse a Python file and extract model class info."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        tree = ast.parse(source)
    except SyntaxError:
        return None

    info = ModelInfo(filepath)

    for node in ast.walk(tree):
        # --- find nn.Module subclasses ---
        if isinstance(node, ast.ClassDef):
            bases = [_get_dotted_name(b) for b in node.bases]
            is_module = any('Module' in b for b in bases)
            if not is_module:
                continue

            # Analyze forward()
            fwd_req, fwd_params, fwd_node = _analyze_method(node, 'forward')
            ret_pattern, ret_count = ('unknown', 0)
            if fwd_node:
                ret_pattern, ret_count = _detect_return_pattern(fwd_node)

            # Analyze __init__()
            init_req, init_params, _ = _analyze_method(node, '__init__')

            info.classes.append({
                'name': node.name,
                'bases': bases,
                'fwd_params': fwd_req,         # REQUIRED params only (no defaults)
                'fwd_all_params': fwd_params,  # all param names
                'ret_pattern': ret_pattern,
                'ret_count': ret_count,
                'init_params': init_params,    # __init__ param names (excl self)
                'init_required': init_req,     # required init params count
            })

        # --- find CONFIGS = { ... } dicts ---
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'CONFIGS':
                    if isinstance(node.value, ast.Dict):
                        keys = []
                        for k in node.value.keys:
                            if isinstance(k, ast.Constant):
                                keys.append(str(k.value))
                        info.configs_dict.append(('CONFIGS', keys))

    # --- find in_chans in config functions ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                tname = _get_dotted_name(target)
                if tname and tname.endswith('in_chans'):
                    if isinstance(node.value, ast.Constant):
                        info.in_chans = node.value.value

    return info if (info.classes or info.configs_dict or info.in_chans is not None) else None


# Directories that are almost certainly NOT the main model code
_SKIP_DIRS = {'baseline', 'baselines', 'baseline_registration_models',
              'docker', 'example', 'examples', 'test', 'tests', '__pycache__'}


def scan_model_dir(model_dir):
    """Recursively scan model directory for Python files with nn.Module classes."""
    results = []
    for dirpath, dirs, files in os.walk(model_dir):
        # Skip directories that are unlikely to contain the main model
        dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DIRS]
        for fname in files:
            if not fname.endswith('.py'):
                continue
            if fname == 'adapter.py':
                continue
            fpath = os.path.join(dirpath, fname)
            info = _scan_file(fpath)
            if info:
                results.append(info)
    return results


# ---------------------------------------------------------------------------
# Decide which model class + file to use
# ---------------------------------------------------------------------------

_COMPONENT_NAMES = {
    'mlp', 'attention', 'windowattention', 'block', 'basiclayer', 'patchembed',
    'patchmerging', 'decoder', 'decoderblock', 'encoder', 'encoderblock',
    'conv2drelu', 'conv3drelu', 'registrationhead', 'spatialtransformer',
    'register_model', 'upsample', 'downsample', 'norm', 'loss', 'grad',
    'ncc', 'ssim', 'dice', 'mind', 'mutual', 'regularizer', 'sinpositional',
}


def _is_likely_toplevel_model(cls_name):
    """Heuristic: skip sub-components, keep top-level model classes."""
    lower = cls_name.lower().replace('_', '')
    if lower in _COMPONENT_NAMES:
        return False
    for comp in _COMPONENT_NAMES:
        if lower == comp:
            return False
    return True


def pick_model(scan_results, model_dir, model_name):
    """Interactively help user pick the right model class from scan results."""

    # Flatten all classes with their file info
    candidates = []
    for info in scan_results:
        relpath = os.path.relpath(info.filepath, model_dir)
        for cls in info.classes:
            label = "%s  (%s)  [fwd: %d input(s), returns %s]" % (
                cls['name'], relpath, cls['fwd_params'], cls['ret_pattern'])
            candidates.append((label, cls, info))

    if not candidates:
        print("  No nn.Module subclasses found. You'll need to write the adapter manually.")
        return None, None

    # Filter to likely top-level model classes
    toplevel = [c for c in candidates if _is_likely_toplevel_model(c[1]['name'])]
    pool = toplevel if toplevel else candidates

    # Prefer classes whose names contain the model name (case-insensitive)
    name_norm = model_name.lower().replace('-', '').replace('_', '')
    name_match = [c for c in pool if name_norm in c[1]['name'].lower().replace('-', '').replace('_', '')]
    if name_match:
        pool = name_match

    # Among matches, prefer 2D if available
    pool_2d = [c for c in pool if '2d' in c[2].filepath.lower() or '2D' in c[2].filepath]
    if pool_2d:
        pool = pool_2d

    # If exactly one candidate, auto-select
    if len(pool) == 1:
        choice = pool[0]
        print("  Auto-detected model class: %s" % choice[0])
        confirm = ask("  Use this?", "y")
        if confirm.lower() in ('y', 'yes'):
            return choice[1], choice[2]

    # If a few, let user pick from the filtered list
    if len(pool) <= 15:
        print("\n  Found %d candidate model classes:" % len(pool))
        labels = [c[0] for c in pool]
        chosen_label = ask_choice("  Pick one:", labels)
        idx = labels.index(chosen_label)
        return pool[idx][1], pool[idx][2]

    # Too many — show only top-level
    print("\n  Found %d candidate classes (showing top-level only):" % len(pool))
    labels = [c[0] for c in pool[:20]]
    chosen_label = ask_choice("  Pick one:", labels)
    idx = labels.index(chosen_label)
    return pool[idx][1], pool[idx][2]


def pick_configs(scan_results, model_dir):
    """Find CONFIGS dictionaries."""
    all_configs = []
    for info in scan_results:
        relpath = os.path.relpath(info.filepath, model_dir)
        for var_name, keys in info.configs_dict:
            all_configs.append((var_name, keys, info, relpath))

    if not all_configs:
        return None, None, None

    # Prefer 2D configs
    configs_2d = [c for c in all_configs if '2d' in c[3].lower() or '2D' in c[3]]
    pool = configs_2d if configs_2d else all_configs

    if len(pool) == 1:
        chosen = pool[0]
    else:
        print("\n  Found multiple CONFIGS dicts:")
        labels = ["%s in %s (variants: %s)" % (c[0], c[3], ', '.join(c[1][:4]) + ('...' if len(c[1]) > 4 else ''))
                  for c in pool]
        chosen_label = ask_choice("  Pick one:", labels)
        idx = labels.index(chosen_label)
        chosen = pool[idx]

    return chosen[0], chosen[1], chosen[2]


# ---------------------------------------------------------------------------
# Compute import path from a file relative to model_dir
# ---------------------------------------------------------------------------

def compute_sys_path_and_import(model_info, cls_info, model_dir):
    """Figure out what sys.path entry and import statement to use."""
    filepath = model_info.filepath
    cls_name = cls_info['name']

    # Walk up from the file to find the deepest directory that is NOT model_dir
    # and where the file can be imported from
    relpath = os.path.relpath(filepath, model_dir)
    parts = relpath.replace('\\', '/').split('/')

    # The file is: parts[-1] (e.g. TransMorph.py)
    # The dirs are: parts[:-1] (e.g. ['RaFD', 'TransMorph2D', 'models'])

    # Strategy: sys.path = the directory such that `from <remaining> import <cls>` works
    # We want the shallowest sys.path that gives a valid import.
    # For "RaFD/TransMorph2D/models/TransMorph.py" from model_dir:
    #   sys.path = model_dir/RaFD/TransMorph2D  ->  from models.TransMorph import TransMorph
    # This matches what the model's own code uses internally.

    module_file = parts[-1].replace('.py', '')
    dir_parts = parts[:-1]

    # Look for __init__.py files to find the package boundary
    # Walk from the file upward until we find a dir without __init__.py
    sys_path_parts = []
    import_parts = list(dir_parts) + [module_file]

    for i in range(len(dir_parts)):
        check_dir = os.path.join(model_dir, *dir_parts[:i + 1])
        if os.path.isfile(os.path.join(check_dir, '__init__.py')):
            # This dir is a package, so the sys.path should be above it
            sys_path_parts = dir_parts[:i]
            import_parts = dir_parts[i:] + [module_file]
            break
    else:
        # No __init__.py found — try: sys.path = parent of first dir containing the file
        if len(dir_parts) >= 1:
            # Check if the deepest dir has __init__.py
            for depth in range(len(dir_parts) - 1, -1, -1):
                check_dir = os.path.join(model_dir, *dir_parts[:depth + 1])
                if os.path.isfile(os.path.join(check_dir, '__init__.py')):
                    sys_path_parts = dir_parts[:depth]
                    import_parts = dir_parts[depth:] + [module_file]
                    break
            else:
                # No packages at all — add the file's parent dir to sys.path
                sys_path_parts = dir_parts
                import_parts = [module_file]

    if sys_path_parts:
        # Use os.path.join at runtime so it works cross-platform
        sys_path_rel = sys_path_parts
    else:
        sys_path_rel = None

    import_module = '.'.join(import_parts)

    return sys_path_rel, import_module, cls_name


# ---------------------------------------------------------------------------
# Detect in_chans across all scan results
# ---------------------------------------------------------------------------

def detect_in_chans(scan_results):
    for info in scan_results:
        if info.in_chans is not None:
            return info.in_chans
    return None


# ---------------------------------------------------------------------------
# Generate adapter.py
# ---------------------------------------------------------------------------

def generate_adapter(name, model_dir, cls_info, model_info, configs_var, configs_keys, configs_info, in_chans):
    """Generate a working adapter.py string."""

    sys_path_rel, import_module, cls_name = compute_sys_path_and_import(model_info, cls_info, model_dir)

    fwd_params = cls_info['fwd_params']     # 1 = concat, 2 = separate
    ret_pattern = cls_info['ret_pattern']   # tuple / list / single

    # Determine if model needs concatenated single-channel input
    concat_input = (fwd_params == 1)
    needs_grayscale = (in_chans is not None and in_chans <= 2 and concat_input)

    # Build sys.path setup
    if sys_path_rel:
        parts_str = ', '.join(repr(p) for p in sys_path_rel)
        sys_path_code = (
            "_sub = os.path.join(os.path.dirname(__file__), %s)\n"
            "if _sub not in sys.path:\n"
            "    sys.path.insert(0, _sub)\n"
        ) % parts_str
    else:
        sys_path_code = (
            "_root = os.path.dirname(__file__)\n"
            "if _root not in sys.path:\n"
            "    sys.path.insert(0, _root)\n"
        )

    # Build import line
    extra_imports = ""
    if configs_var and configs_info and configs_info.filepath == model_info.filepath:
        import_line = "from %s import %s, %s" % (import_module, cls_name, configs_var)
    elif configs_var and configs_info:
        # CONFIGS in a different file than the class
        cfg_sys_rel, cfg_import_mod, _ = compute_sys_path_and_import(
            configs_info, {'name': configs_var}, model_dir)
        import_line = "from %s import %s" % (import_module, cls_name)
        extra_imports = "from %s import %s" % (cfg_import_mod, configs_var)
    else:
        import_line = "from %s import %s" % (import_module, cls_name)

    # Build build() function
    init_params = cls_info.get('init_params', [])
    if configs_var and configs_keys:
        default_variant = configs_keys[0]
        build_body = (
            '    variant = getattr(args, \'%s_variant\', \'%s\')\n'
            '    config = %s[variant]\n'
            '    config.img_size = tuple(args.image_size)\n'
            '    net = %s(config)\n'
            '    return nn.DataParallel(net, device_ids=args.gpus)'
        ) % (name.replace('-', '_'), default_variant, configs_var, cls_name)
    elif init_params == ['args'] or init_params == ['opt']:
        # Model takes a single namespace (like RAFT)
        build_body = (
            '    net = %s(args)\n'
            '    return nn.DataParallel(net, device_ids=args.gpus)'
        ) % cls_name
    elif len(init_params) == 1 and init_params[0] in ('config', 'cfg', 'hparams'):
        build_body = (
            '    net = %s(args)  # pass args as config\n'
            '    return nn.DataParallel(net, device_ids=args.gpus)'
        ) % cls_name
    else:
        build_body = (
            '    net = %s()  # TODO: check constructor args — __init__ takes: %s\n'
            '    return nn.DataParallel(net, device_ids=args.gpus)'
        ) % (cls_name, ', '.join(init_params) if init_params else 'none')

    # Build forward() function
    torch_import = ""
    if concat_input:
        torch_import = "import torch\n"

    # Detect normalization: registration models expect [0,1], optical flow models [0,255].
    registration_keywords = ('morph', 'register', 'deform', 'warp', 'stn')
    is_registration = any(kw in name.lower() for kw in registration_keywords)
    normalize = (in_chans is not None) or concat_input or is_registration

    if concat_input and needs_grayscale:
        fwd_prep = (
            "    gray1 = img1.mean(dim=1, keepdim=True) / 255.0\n"
            "    gray2 = img2.mean(dim=1, keepdim=True) / 255.0\n"
            "    x_in = torch.cat((gray1, gray2), dim=1)\n"
        )
        fwd_call = "    out = model(x_in)"
    elif concat_input:
        fwd_prep = (
            "    x_in = torch.cat((img1 / 255.0, img2 / 255.0), dim=1)\n"
        )
        fwd_call = "    out = model(x_in)"
    elif normalize:
        fwd_prep = ""
        fwd_call = "    out = model(img1 / 255.0, img2 / 255.0)"
    else:
        fwd_prep = ""
        fwd_call = "    out = model(img1, img2)"

    if ret_pattern == 'tuple':
        fwd_return = "    warped, flow = out\n    return [flow]"
    elif ret_pattern == 'list':
        fwd_return = "    return out if isinstance(out, list) else [out]"
    else:
        fwd_return = "    return [out] if not isinstance(out, (list, tuple)) else list(out)"

    # Assemble
    lines = []
    lines.append('"""%s adapter — auto-generated by create_adapter.py."""\n' % name)
    lines.append('import sys')
    lines.append('import os\n')
    if torch_import:
        lines.append(torch_import.rstrip())
    lines.append('import torch.nn as nn\n')
    lines.append(sys_path_code)
    lines.append(import_line)
    if extra_imports:
        lines.append(extra_imports)
    lines.append('\n')

    lines.append('def build(args):')
    lines.append('    """Return a %s model wrapped in DataParallel."""' % name)
    lines.append(build_body)
    lines.append('\n')

    lines.append('def forward(model, img1, img2, args):')
    lines.append('    """Run %s and return a list of flow predictions [(B,2,H,W), ...]."""' % name)
    if fwd_prep:
        lines.append(fwd_prep.rstrip())
    lines.append(fwd_call)
    lines.append(fwd_return)
    lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Generate config YAML
# ---------------------------------------------------------------------------

def generate_config(name, configs_keys, in_chans):
    """Generate a config YAML string."""

    # Copy data_root from existing config
    data_root_block = '  - "path/to/your/data"'
    for ref in ('config.yaml', 'raft.yaml', 'voxelmorph.yaml'):
        ref_path = os.path.join(ROOT, 'config', ref)
        if os.path.isfile(ref_path):
            try:
                import yaml
                with open(ref_path, 'r') as f:
                    ref_cfg = yaml.safe_load(f)
                if ref_cfg and ref_cfg.get('data_root'):
                    lines = []
                    for p in ref_cfg['data_root']:
                        lines.append('  - "%s"' % p.replace('\\', '\\\\'))
                    data_root_block = '\n'.join(lines)
                    break
            except Exception:
                pass

    is_transformer = any(kw in name.lower() for kw in ('trans', 'swin', 'vit', 'attention'))
    batch_default = 4 if is_transformer else 12
    img_size = "[256, 256]" if is_transformer else "[384, 512]"

    extra_yaml = ""
    if configs_keys:
        extra_yaml = "\n# ── %s-specific ─────────────────────────────────────\n" % name.capitalize()
        extra_yaml += "# Variants: %s\n" % ', '.join(configs_keys)
        extra_yaml += "%s_variant: %s\n" % (name.replace('-', '_'), configs_keys[0])

    content = textwrap.dedent("""\
        # ── General ──────────────────────────────────────────────
        name: experiment
        model: {name}
        loss: ncc
        restore_ckpt: null
        gpus: [0]
        mixed_precision: false

        # ── Training ─────────────────────────────────────────────
        lr: 0.0001
        num_steps: 100000
        batch_size: {batch}
        image_size: {img_size}
        clip: 1.0
        wdecay: 0.00005
        epsilon: 1.0e-8
        add_noise: false

        # ── Loss ─────────────────────────────────────────────────
        smooth_weight: 0.02
        mag_weight: 0.0

        # ── Dataset ──────────────────────────────────────────────
        data_root:
        {data_root_block}
        mat_pattern: "envelope_flow*.mat"
        val_ratio: 0.1
        log_compress: false
        cache_max: 0
        num_workers: 8
        no_persistent_workers: false

        # ── Checkpoints & Visualization ──────────────────────────
        checkpoint_freq: 5000
        vis_freq: 1000
    """).format(name=name, batch=batch_default, img_size=img_size, data_root_block=data_root_block)

    content += extra_yaml
    return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Onboard a new model — scan code, generate adapter + config, setup env")
    parser.add_argument('--name', type=str, default=None, help='model folder name in models/')
    parser.add_argument('--setup', action='store_true', help='also run setup_model.ps1 to create conda env')
    parser.add_argument('--force', action='store_true', help='overwrite existing adapter.py and config')
    cli = parser.parse_args()

    print("=" * 60)
    print("  Model Onboarding")
    print("=" * 60)

    # ── Step 1: model name ────────────────────────────────────────────────
    name = cli.name or ask("Model name (folder in models/)")
    if not name:
        print("Error: model name is required.")
        sys.exit(1)

    model_dir = os.path.join(ROOT, 'models', name)
    if not os.path.isdir(model_dir):
        print("Error: models/%s/ not found. Clone the model there first." % name)
        sys.exit(1)

    print("\n  Scanning models/%s/ ..." % name)

    # ── Step 2: scan code ─────────────────────────────────────────────────
    scan_results = scan_model_dir(model_dir)
    total_classes = sum(len(info.classes) for info in scan_results)
    total_configs = sum(len(info.configs_dict) for info in scan_results)
    print("  Found %d nn.Module classes across %d files" % (total_classes, len(scan_results)))
    if total_configs:
        print("  Found %d CONFIGS dict(s)" % total_configs)

    # ── Step 3: find CONFIGS (narrows the search) ──────────────────────
    configs_var, configs_keys, configs_info = pick_configs(scan_results, model_dir)
    if configs_keys:
        print("  CONFIGS variants: %s" % ', '.join(configs_keys))

    # ── Step 4: pick model class ──────────────────────────────────────────
    # If we found a CONFIGS dict, look for the model class in the SAME file first
    cls_info, model_info = None, None
    if configs_info:
        same_file_classes = [c for c in configs_info.classes
                            if _is_likely_toplevel_model(c['name'])]
        name_norm = name.lower().replace('-', '').replace('_', '')
        matched = [c for c in same_file_classes
                   if name_norm in c['name'].lower().replace('-', '').replace('_', '')]
        if len(matched) == 1:
            cls_info, model_info = matched[0], configs_info
            relpath = os.path.relpath(configs_info.filepath, model_dir)
            print("  Auto-detected model class: %s  (%s)" % (cls_info['name'], relpath))

    if cls_info is None:
        cls_info, model_info = pick_model(scan_results, model_dir, name)
    if cls_info is None:
        sys.exit(1)

    # ── Step 5: detect in_chans ───────────────────────────────────────────
    in_chans = detect_in_chans(scan_results)
    if in_chans:
        print("  Detected in_chans = %d" % in_chans)

    # ── Step 6: generate adapter.py ───────────────────────────────────────
    adapter_path = os.path.join(model_dir, 'adapter.py')
    if os.path.isfile(adapter_path) and not cli.force:
        overwrite = ask("\n  adapter.py already exists — overwrite? (y/n)", "n")
        if overwrite.lower() not in ('y', 'yes'):
            print("  Skipped adapter.py")
            adapter_path = None

    if adapter_path:
        adapter_code = generate_adapter(
            name, model_dir, cls_info, model_info,
            configs_var, configs_keys, configs_info, in_chans,
        )
        with open(adapter_path, 'w', encoding='utf-8') as f:
            f.write(adapter_code)
        print("\n  Created: models/%s/adapter.py" % name)

    # ── Step 7: generate config YAML ──────────────────────────────────────
    config_path = os.path.join(ROOT, 'config', '%s.yaml' % name)
    if os.path.isfile(config_path) and not cli.force:
        overwrite = ask("  config/%s.yaml already exists — overwrite? (y/n)" % name, "n")
        if overwrite.lower() not in ('y', 'yes'):
            print("  Skipped config/%s.yaml" % name)
            config_path = None

    if config_path:
        os.makedirs(os.path.join(ROOT, 'config'), exist_ok=True)
        config_content = generate_config(name, configs_keys, in_chans)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print("  Created: config/%s.yaml" % name)

    # ── Step 8: optionally run setup_model.ps1 ────────────────────────────
    setup_script = os.path.join(ROOT, 'setup_model.ps1')
    if cli.setup and os.path.isfile(setup_script):
        print("\n  Running setup_model.ps1 for %s ..." % name)
        subprocess.run(
            ['powershell', '-ExecutionPolicy', 'Bypass', '-File', setup_script, name],
            cwd=ROOT,
        )
    elif os.path.isfile(setup_script):
        run_setup = ask("\n  Run setup_model.ps1 to create conda env? (y/n)", "y")
        if run_setup.lower() in ('y', 'yes'):
            print("  Running setup_model.ps1 for %s ..." % name)
            subprocess.run(
                ['powershell', '-ExecutionPolicy', 'Bypass', '-File', setup_script, name],
                cwd=ROOT,
            )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Done! To train:")
    print("=" * 60)
    print("    conda activate %s" % name)
    print("    python train.py --config config/%s.yaml" % name)
    print()


if __name__ == '__main__':
    main()
