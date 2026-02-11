"""
Dataset and dataloader. Reads data: and forward.input from config.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import platform, re

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _load_mat_key(path, key):
    mat = sio.loadmat(path, simplify_cells=False)
    if key not in mat:
        raise KeyError(f"'{key}' not in {path}. Keys: {[k for k in mat if not k.startswith('__')]}")
    arr = np.asarray(mat[key])
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D for '{key}' in {path}, got {arr.shape}")
    return arr

def _normalize(arr, mode):
    if mode == "none": return arr
    if mode == "per_frame":
        mn, mx = float(arr.min()), float(arr.max())
        return (arr - mn) / (mx - mn) if mx > mn else arr
    if mode == "global_minmax":
        return np.clip(arr, 0, 255) / 255.0
    raise ValueError(f"Unknown data.normalization: '{mode}'")

def _gray_to_rgb(x):
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x

def _convert(img1, img2, mode):
    m = mode.lower()
    if m == "pair_gray":  return {"img1": img1, "img2": img2}
    if m == "pair_rgb":   return {"img1": _gray_to_rgb(img1), "img2": _gray_to_rgb(img2)}
    if m == "stack2ch":   return {"x": torch.cat([img1, img2], dim=0)}
    if m == "stack6ch":   return {"x": torch.cat([_gray_to_rgb(img1), _gray_to_rgb(img2)], dim=0)}
    if m == "video_tchw": return {"x": torch.stack([img1, img2], dim=0)}
    raise ValueError(f"Unknown forward.input: '{mode}'")


class PairDataset(Dataset):
    def __init__(self, pairs, key, resize_hw, normalization, input_mode):
        self.pairs = pairs
        self.key = key
        self.resize_hw = resize_hw
        self.norm = normalization
        self.input_mode = input_mode

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2 = self.pairs[idx]
        a1 = _normalize(_load_mat_key(p1, self.key).astype(np.float32), self.norm)
        a2 = _normalize(_load_mat_key(p2, self.key).astype(np.float32), self.norm)

        img1 = torch.from_numpy(a1).unsqueeze(0)
        img2 = torch.from_numpy(a2).unsqueeze(0)

        if self.resize_hw:
            H, W = self.resize_hw
            img1 = F.interpolate(img1.unsqueeze(0), (H, W), mode="bilinear", align_corners=False).squeeze(0)
            img2 = F.interpolate(img2.unsqueeze(0), (H, W), mode="bilinear", align_corners=False).squeeze(0)

        sample = _convert(img1, img2, self.input_mode)
        if "img1" not in sample:
            sample["img1"] = img1
            sample["img2"] = img2
        sample["meta"] = {"p1": str(p1), "p2": str(p2), "pair_idx": idx}
        return sample


def _resolve_pairs(cfg: dict, repo_root: Path):
    """Build the list of (file1, file2) pairs from config."""
    data = cfg["data"]
    root = Path(data["path"])
    if not root.is_absolute():
        root = (repo_root / root).resolve()

    files = sorted(root.glob(data.get("pattern", "*.mat")), key=lambda p: _natural_key(p.name))
    if len(files) < 2:
        raise RuntimeError(f"Need >=2 files in {root}, found {len(files)}")

    sel = data.get("select") or {}
    fr = sel.get("file_range")
    if fr:
        files = files[int(fr[0]) - 1 : int(fr[1])]

    pairs = [(files[i].resolve(), files[i+1].resolve()) for i in range(len(files) - 1)]

    pi = sel.get("pair_indices")
    if pi:
        pairs = [pairs[int(i)] for i in pi]
    mp = sel.get("max_pairs")
    if mp:
        pairs = pairs[:int(mp)]

    return pairs


def _make_loader(pairs, cfg, repo_root, shuffle=True):
    """Create a DataLoader from a list of pairs."""
    data = cfg["data"]
    fwd = cfg.get("forward", {})
    train = cfg.get("training", {})
    runtime = cfg.get("runtime", {})

    resize_hw = None
    rs = data.get("resize")
    if rs:
        resize_hw = (int(rs[0]), int(rs[1]))

    ds = PairDataset(
        pairs=pairs,
        key=data.get("key", "img"),
        resize_hw=resize_hw,
        normalization=str(data.get("normalization", "none")),
        input_mode=fwd.get("input", "pair_gray"),
    )

    batch_size = int(train.get("batch_size", 4))
    num_workers = int(runtime.get("num_workers", 0))
    if platform.system().lower() == "darwin":
        num_workers = 0

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False, drop_last=True)


def build_dataloader(cfg: dict, repo_root: Path) -> DataLoader:
    """Build a single dataloader with all pairs (used by inference and legacy code)."""
    pairs = _resolve_pairs(cfg, repo_root)
    is_inference = str(cfg.get("runtime", {}).get("mode", "train")).lower() == "inference"
    return _make_loader(pairs, cfg, repo_root, shuffle=not is_inference)


def build_train_val_loaders(cfg: dict, repo_root: Path):
    """
    Build train and (optionally) val dataloaders based on training.validation_split.

    Returns (train_loader, val_loader).  val_loader is None when split <= 0.
    """
    import random

    pairs = _resolve_pairs(cfg, repo_root)
    tcfg = cfg.get("training", {})
    val_split = float(tcfg.get("validation_split", 0))

    if val_split <= 0 or val_split >= 1:
        return _make_loader(pairs, cfg, repo_root, shuffle=True), None

    # Deterministic shuffle so train/val split is reproducible
    rng = random.Random(42)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)

    n_val = max(1, int(len(pairs) * val_split))
    val_idx = set(indices[:n_val])

    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    val_pairs   = [pairs[i] for i in range(len(pairs)) if i in val_idx]

    train_loader = _make_loader(train_pairs, cfg, repo_root, shuffle=True)
    val_loader   = _make_loader(val_pairs, cfg, repo_root, shuffle=False)

    return train_loader, val_loader
