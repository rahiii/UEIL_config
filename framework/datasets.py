# framework/datasets.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import re
import platform

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import scipy.io as sio


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _resolve_root(repo_root: Path, root_str: str) -> Path:
    p = Path(root_str)
    return p if p.is_absolute() else (repo_root / p).resolve()


def _load_mat_key(mat_path: Path, key: str) -> np.ndarray:
    mat = sio.loadmat(mat_path, simplify_cells=False)
    if key not in mat:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {mat_path}. Available keys: {keys}")
    arr = np.asarray(mat[key])
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for key '{key}' in {mat_path}, got shape {arr.shape}")
    return arr


def _apply_normalization(arr: np.ndarray, norm_cfg: Dict[str, Any]) -> np.ndarray:
    ntype = str(norm_cfg.get("type", "none")).lower()
    clip_min = float(norm_cfg.get("clip_min", 0.0))
    clip_max = float(norm_cfg.get("clip_max", 255.0))

    if ntype == "none":
        return arr

    if ntype == "per_frame":
        mn = float(arr.min())
        mx = float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        return arr

    if ntype == "global_minmax":
        arr = np.clip(arr, clip_min, clip_max)
        arr = (arr - clip_min) / (clip_max - clip_min + 1e-6)
        return arr

    raise ValueError(f"Unknown normalization.type: {ntype}")


def _resize_tensor_chw(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x4 = x.unsqueeze(0)
    x4r = F.interpolate(x4, size=(height, width), mode="bilinear", align_corners=False)
    return x4r.squeeze(0)


# Conversion helpers for model-agnostic data views
def _to_dtype(x: torch.Tensor, dtype: str) -> torch.Tensor:
    dtype = dtype.lower()
    if dtype == "float32":
        return x.float()
    if dtype == "float16":
        return x.half()
    raise ValueError(f"Unsupported dtype: {dtype}")


def _ensure_chw(x: torch.Tensor) -> torch.Tensor:
    # Expecting [1,H,W] already; keep as-is, but validate
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")
    return x


def _repeat_gray_to_rgb(x: torch.Tensor) -> torch.Tensor:
    # x: [1,H,W] -> [3,H,W]
    x = _ensure_chw(x)
    if x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    if x.shape[0] == 3:
        return x
    raise ValueError(f"Expected C=1 or C=3, got C={x.shape[0]}")


def _stack_pair_channels(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    # [1,H,W] + [1,H,W] -> [2,H,W]
    img1 = _ensure_chw(img1)
    img2 = _ensure_chw(img2)
    return torch.cat([img1, img2], dim=0)


def _stack_pair_rgb6(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    # [1,H,W] -> [3,H,W] for each, then concat -> [6,H,W]
    a = _repeat_gray_to_rgb(img1)
    b = _repeat_gray_to_rgb(img2)
    return torch.cat([a, b], dim=0)


def _pair_to_video_tchw(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    # [1,H,W],[1,H,W] -> [T=2,C=1,H,W]
    img1 = _ensure_chw(img1)
    img2 = _ensure_chw(img2)
    return torch.stack([img1, img2], dim=0)


def apply_conversion(
    img1: torch.Tensor,
    img2: torch.Tensor,
    conv_cfg: Dict[str, Any] | None
) -> Dict[str, torch.Tensor]:
    """
    Returns either:
      - {"img1": ..., "img2": ...}  (pair views)
      - {"x": ...}                  (stacked/single input views)
    """
    conv_cfg = conv_cfg or {}
    view = str(conv_cfg.get("view", "pair_gray")).lower()
    out_dtype = str(conv_cfg.get("dtype", "float32"))

    if view == "pair_gray":
        return {"img1": _to_dtype(img1, out_dtype), "img2": _to_dtype(img2, out_dtype)}

    if view == "pair_rgb":
        return {
            "img1": _to_dtype(_repeat_gray_to_rgb(img1), out_dtype),
            "img2": _to_dtype(_repeat_gray_to_rgb(img2), out_dtype),
        }

    if view == "stack2ch":
        x = _stack_pair_channels(img1, img2)       # [2,H,W]
        return {"x": _to_dtype(x, out_dtype)}

    if view == "stack6ch":
        x = _stack_pair_rgb6(img1, img2)           # [6,H,W]
        return {"x": _to_dtype(x, out_dtype)}

    if view == "video_tchw":
        x = _pair_to_video_tchw(img1, img2)        # [2,1,H,W]
        return {"x": _to_dtype(x, out_dtype)}

    raise ValueError(f"Unknown conversion.view: {view}")


@dataclass
class MatConsecutivePairConfig:
    repo_root: Path
    root: str
    pattern: str
    key: str = "img"
    resize_hw: Optional[Tuple[int, int]] = None
    normalization: Optional[Dict[str, Any]] = None
    dtype: str = "float32"

    # new
    file_range: Optional[Tuple[int, int]] = None      # 1-based inclusive file indices
    pair_indices: Optional[List[int]] = None          # 0-based indices over pairs
    max_pairs: Optional[int] = None
    conversion: Optional[Dict[str, Any]] = None       # conversion config for model-agnostic views


class MatConsecutivePairDataset(Dataset):
    def __init__(self, cfg: MatConsecutivePairConfig):
        self.cfg = cfg
        self.root = _resolve_root(cfg.repo_root, cfg.root)
        if not self.root.exists():
            raise FileNotFoundError(f"Data root not found: {self.root}")

        files = sorted(self.root.glob(cfg.pattern), key=lambda p: _natural_key(p.name))
        if len(files) < 2:
            raise RuntimeError(f"Need at least 2 .mat files. Found {len(files)} for pattern '{cfg.pattern}' in {self.root}")

        # file_range selection (1-based inclusive)
        if cfg.file_range is not None:
            a, b = cfg.file_range
            if a < 1 or b < 1 or a > b:
                raise ValueError(f"Invalid file_range: {cfg.file_range}. Must be [start,end] with start>=1 and start<=end.")
            a0 = a - 1
            b0 = b - 1
            if b0 >= len(files):
                raise ValueError(f"file_range end={b} exceeds number of files={len(files)}")
            files = files[a0:b0 + 1]

        all_pairs: List[Tuple[Path, Path]] = [
            (files[i].resolve(), files[i + 1].resolve())
            for i in range(len(files) - 1)
        ]

        # explicit pair selection (0-based over pairs)
        if cfg.pair_indices is not None:
            pairs = []
            n = len(all_pairs)
            for idx in cfg.pair_indices:
                if idx < 0 or idx >= n:
                    raise ValueError(f"pair_indices contains {idx}, but valid range is [0, {n-1}]")
                pairs.append(all_pairs[idx])
            all_pairs = pairs

        # cap
        if cfg.max_pairs is not None:
            all_pairs = all_pairs[: int(cfg.max_pairs)]

        self.pairs = all_pairs

        if cfg.dtype == "float32":
            self.out_dtype = np.float32
        elif cfg.dtype == "float16":
            self.out_dtype = np.float16
        else:
            raise ValueError(f"Unsupported dtype: {cfg.dtype}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p1, p2 = self.pairs[idx]

        a1 = _load_mat_key(p1, self.cfg.key).astype(self.out_dtype, copy=False)
        a2 = _load_mat_key(p2, self.cfg.key).astype(self.out_dtype, copy=False)

        norm_cfg = self.cfg.normalization or {"type": "none"}
        a1 = _apply_normalization(a1, norm_cfg)
        a2 = _apply_normalization(a2, norm_cfg)

        img1 = torch.from_numpy(a1).unsqueeze(0)  # [1,H,W]
        img2 = torch.from_numpy(a2).unsqueeze(0)

        if self.cfg.resize_hw is not None:
            H, W = self.cfg.resize_hw
            img1 = _resize_tensor_chw(img1, H, W)
            img2 = _resize_tensor_chw(img2, H, W)

        # Apply conversion pipeline (model-agnostic views)
        conv_out = apply_conversion(img1, img2, self.cfg.conversion)

        sample = {"meta": {"p1": str(p1), "p2": str(p2), "key": self.cfg.key, "pair_idx": idx}}

        # If conversion returned x, use that; else keep img1/img2
        sample.update(conv_out)

        # Always keep canonical fields too (optional but useful)
        # If you want strict minimal output, remove these two lines
        if "img1" not in sample:
            sample["img1"] = img1.float()
        if "img2" not in sample:
            sample["img2"] = img2.float()

        return sample


def build_dataloader(cfg: Dict[str, Any], repo_root: Path) -> DataLoader:
    din = cfg["input"]["data_input"]
    norm = cfg.get("input", {}).get("normalization", {"type": "none"})
    conv = cfg.get("input", {}).get("conversion", {})
    sel = din.get("select", {}) or {}

    root = str(din["path"])
    pattern = str(din.get("image_type", "*.mat"))
    key = str(din.get("image_key", "img"))

    resize = din.get("image_resize", None)
    resize_hw = None
    if resize and resize.get("height") and resize.get("width"):
        resize_hw = (int(resize["height"]), int(resize["width"]))

    runtime = cfg.get("runtime", {})
    model_name = str(cfg.get("model", "raft")).lower()
    train_cfg = cfg.get("training", {}).get(model_name, {})

    batch_size = int(train_cfg.get("batch_size", 4))
    shuffle = bool(train_cfg.get("shuffle", True))  # allow deterministic inference/export by config
    num_workers = int(runtime.get("num_workers", 0))
    device = str(runtime.get("device", "auto")).lower()

    is_mac = platform.system().lower() == "darwin"
    if is_mac and device in ("mps", "auto"):
        num_workers = 0

    file_range = sel.get("file_range", None)
    if file_range is not None:
        file_range = (int(file_range[0]), int(file_range[1]))

    pair_indices = sel.get("pair_indices", None)
    if pair_indices is not None:
        pair_indices = [int(x) for x in pair_indices]

    max_pairs = sel.get("max_pairs", None)
    if max_pairs is not None:
        max_pairs = int(max_pairs)

    ds = MatConsecutivePairDataset(
        MatConsecutivePairConfig(
            repo_root=repo_root,
            root=root,
            pattern=pattern,
            key=key,
            resize_hw=resize_hw,
            normalization=norm,
            dtype="float32",
            file_range=file_range,
            pair_indices=pair_indices,
            max_pairs=max_pairs,
            conversion=conv if conv else None,
        )
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        drop_last=True,
    )
