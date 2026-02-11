"""
Config-driven model adapter.

Reads model: and forward: from the YAML config to import, build, and call any model.

Config → code traceability
--------------------------
  model.entry_point          → _import_class()
  model.args                 → build()  kwargs or Namespace attrs
  model.args_mode            → build()  "kwargs" (default) or "namespace"
  forward.input              → run()    which batch keys to pass
  forward.call_kwargs        → run()    extra kwargs to model() e.g. {iters: 12}
  forward.call_kwargs_test   → run()    override kwargs during inference
  forward.output             → _extract_flow()
  forward.output_format      → _normalize_flow()

Escape hatch: create models/<n>/adapter.py with class Adapter that has
build(cfg) and run(model, batch, cfg). The framework checks for this first.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


# ── Helpers ─────────────────────────────────────────────────────────────


def _import_class(dotted_path: str):
    mod_path, cls_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    if not hasattr(mod, cls_name):
        avail = [a for a in dir(mod) if not a.startswith("_")]
        raise ImportError(f"{mod_path} has no '{cls_name}'. Available: {avail}")
    return getattr(mod, cls_name)


def _extract_flow(raw, mode: str):
    m = mode.lower().strip()
    if m == "direct":
        # Many models return a tuple/list of multi-scale flows during
        # training but a single tensor at eval.  Handle both gracefully.
        if isinstance(raw, (tuple, list)):
            return raw[0]
        return raw
    if m == "list_last":    return raw[-1]
    if m == "list_first":   return raw[0]
    if m == "tuple_first":  return raw[0]
    if m.startswith("tuple_index:"):
        return raw[int(m.split(":", 1)[1])]
    if m.startswith("dict_key:"):
        return raw[m.split(":", 1)[1]]
    raise ValueError(f"Unknown forward.output: '{mode}'")


def _normalize_flow(flow, fmt: str) -> torch.Tensor:
    if flow.ndim == 3:
        flow = flow.unsqueeze(0)
    f = fmt.lower().strip()
    if f == "flow_2hw":  return flow
    if f == "flow_hw2":  return flow.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Unknown forward.output_format: '{fmt}'")


def _flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    fx = flow[:, 0].detach().cpu().numpy()
    fy = flow[:, 1].detach().cpu().numpy()
    B, H, W = fx.shape
    out = np.zeros((B, 3, H, W), dtype=np.uint8)
    for b in range(B):
        mag = np.sqrt(fx[b]**2 + fy[b]**2)
        ang = np.arctan2(fy[b], fx[b])
        mx = mag.max() + 1e-7
        h = ((ang + np.pi) / (2 * np.pi)) * 6.0
        v = mag / mx
        i = np.floor(h).astype(np.int32) % 6
        f = h - np.floor(h)
        p = np.zeros_like(v)
        q = v * (1.0 - f)
        t = v * f
        r, g, bl = np.zeros_like(v), np.zeros_like(v), np.zeros_like(v)
        for sec, (rv, gv, bv) in enumerate(
            [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]
        ):
            mask = i == sec
            r[mask], g[mask], bl[mask] = rv[mask], gv[mask], bv[mask]
        out[b,0] = (r*255).astype(np.uint8)
        out[b,1] = (g*255).astype(np.uint8)
        out[b,2] = (bl*255).astype(np.uint8)
    return torch.from_numpy(out)


# ── Generic Adapter ────────────────────────────────────────────────────


class GenericAdapter:

    def __init__(self, cfg: dict, repo_root: Path):
        mcfg = cfg["model"]
        fwd = cfg.get("forward", {})

        self.entry_point = mcfg["entry_point"]
        self.model_args = mcfg.get("args") or {}
        self.args_mode = mcfg.get("args_mode", "kwargs")  # "kwargs" or "namespace"

        self.input_mode = fwd.get("input", "pair_rgb")
        self.output_mode = fwd.get("output", "direct")
        self.output_format = fwd.get("output_format", "flow_2hw")
        self.call_kwargs = fwd.get("call_kwargs") or {}
        self.call_kwargs_test = fwd.get("call_kwargs_test") or {}

        # Make model dirs importable
        for p in [repo_root, repo_root / "models"]:
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))

        # Add the model's own directory + core/ to path
        name = mcfg["name"]
        model_dir = repo_root / "models" / name
        for sub in [model_dir, model_dir / "core"]:
            if sub.is_dir() and str(sub) not in sys.path:
                sys.path.insert(0, str(sub))

        # Handle model.sys_path for repos with nested internal imports.
        # When a model file does e.g. `import models.configs_X`, we need
        # the correct subdirectory on sys.path so that `models/` resolves
        # to the repo's own sub-package, not the top-level models/ dir.
        sys_path_rel = mcfg.get("sys_path", "")
        if sys_path_rel:
            sub_dir = model_dir / sys_path_rel
            if sub_dir.is_dir() and str(sub_dir) not in sys.path:
                sys.path.insert(0, str(sub_dir))

    def build(self, cfg: dict) -> torch.nn.Module:
        cls = _import_class(self.entry_point)

        if self.args_mode == "namespace":
            # Research models often expect argparse.Namespace
            args = argparse.Namespace(**self.model_args)
            return cls(args)
        else:
            return cls(**self.model_args)

    def run(self, model: torch.nn.Module, batch: dict, cfg: dict) -> dict:
        device = next(model.parameters()).device
        inp = self.input_mode.lower()

        # Pick call_kwargs based on train/eval
        if model.training:
            extra = dict(self.call_kwargs)
        else:
            extra = dict(self.call_kwargs)
            extra.update(self.call_kwargs_test)  # test overrides train

        # Call model
        if inp in ("pair_gray", "pair_rgb"):
            raw = model(batch["img1"].to(device), batch["img2"].to(device), **extra)
        elif inp in ("stack2ch", "stack6ch", "video_tchw"):
            raw = model(batch["x"].to(device), **extra)
        else:
            raise ValueError(f"Unknown forward.input: '{self.input_mode}'")

        flow_raw = _extract_flow(raw, self.output_mode)
        flow = _normalize_flow(flow_raw, self.output_format)

        return {
            "disp_x": flow[:, 0].detach().cpu(),
            "disp_y": flow[:, 1].detach().cpu(),
            "flow": flow,
            "flow_vis_rgb": _flow_to_rgb(flow),
        }


# ── Loader ──────────────────────────────────────────────────────────────


def load_adapter(cfg: dict, repo_root: Path):
    """
    1. If models/<n>/adapter.py exists → use it (escape hatch)
    2. Otherwise → GenericAdapter driven by config

    Uses file-path import for the custom adapter check so that we don't
    cache the top-level ``models/`` package in ``sys.modules`` (which
    would conflict with models that have their own internal ``models/``
    sub-package, e.g. ``models/<name>/subdir/models/``).
    """
    name = cfg["model"]["name"]

    # Check for custom adapter (file-path based, no package pollution)
    adapter_py = repo_root / "models" / name / "adapter.py"
    if adapter_py.exists():
        spec = importlib.util.spec_from_file_location(
            f"_custom_adapter_{name}", str(adapter_py),
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception:
                mod = None
            if mod:
                for cls_name in [
                    "Adapter",
                    f"{name.upper()}Adapter",
                    f"{name.capitalize()}Adapter",
                ]:
                    if hasattr(mod, cls_name):
                        return getattr(mod, cls_name)()

    return GenericAdapter(cfg, repo_root)
