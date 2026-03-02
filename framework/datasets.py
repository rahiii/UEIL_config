import os
import re
import random
from collections import OrderedDict
from glob import glob
import os.path as osp

import cv2
import numpy as np
import torch
import torch.utils.data as data

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


def _natural_sort_key(path):
    """Sort by numbers in filename: envelope_flow1, 2, 3, ..., 10, 11."""
    parts = re.split(r'(\d+)', osp.basename(path))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _read_mat_frame(path):
    """Load a single .mat file and return a 2D float32 array."""
    if loadmat is None:
        raise ImportError("scipy is required for .mat support: pip install scipy")
    mat = loadmat(path)
    skip = {'__header__', '__version__', '__globals__'}
    for key in ('img', 'envelope', 'data', 'im', 'frame', 'I'):
        if key in mat:
            arr = np.asarray(mat[key]).squeeze()
            if arr.ndim == 2:
                return arr.astype(np.float32)
    for key in mat:
        if key in skip:
            continue
        arr = np.asarray(mat[key]).squeeze()
        if arr.ndim == 2:
            return arr.astype(np.float32)
    raise ValueError("No 2D array found in %s. Keys: %s" % (path, list(mat.keys())))


class EnvelopeMatDataset(data.Dataset):
    """Loads consecutive frame pairs from envelope .mat files.

    Supports multiple root directories (pairs never cross folders),
    per-frame normalization to 0-255, optional log compression,
    and LRU caching.
    """

    def __init__(self, roots, pattern='envelope_flow*.mat',
                 target_size=(256, 256), split='train', val_ratio=0.1,
                 log_compress=False, cache_max=0, verbose=True):
        super().__init__()
        if isinstance(roots, str):
            roots = [roots]
        self.roots = [osp.abspath(r) for r in roots]
        self.target_size = tuple(target_size)
        self.log_compress = log_compress
        self.cache_max = cache_max
        self._cache = OrderedDict()
        self.init_seed = False

        all_pairs = []
        for r in self.roots:
            files = sorted(glob(osp.join(r, pattern)), key=_natural_sort_key)
            pairs = [[files[i], files[i + 1]] for i in range(len(files) - 1)]
            if pairs and verbose:
                print("  folder %s: %d files -> %d pairs" % (osp.basename(r), len(files), len(pairs)))
            all_pairs.extend(pairs)

        assert len(all_pairs) > 0, (
            "No pairs found matching '%s' in %s" % (pattern, self.roots))

        random.seed(42)
        random.shuffle(all_pairs)
        n_val = max(1, int(len(all_pairs) * val_ratio)) if val_ratio > 0 else 0
        if split == 'train':
            self.pairs = all_pairs[:len(all_pairs) - n_val]
        else:
            self.pairs = all_pairs[len(all_pairs) - n_val:]
        assert len(self.pairs) > 0, "Empty %s split" % split

        if split == 'train' and len(self.pairs) > 20000 and self.cache_max == 0:
            self.cache_max = 5000

    def _get_frame(self, path):
        if path in self._cache:
            if self.cache_max > 0:
                self._cache.move_to_end(path)
            return np.copy(self._cache[path])
        arr = _read_mat_frame(path)
        if self.cache_max > 0:
            while len(self._cache) >= self.cache_max:
                self._cache.popitem(last=False)
        self._cache[path] = arr
        return np.copy(arr)

    def _normalize(self, img):
        if self.log_compress:
            img = np.log1p(np.maximum(img, 0).astype(np.float64)).astype(np.float32)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo) * 255.0
        else:
            img = np.zeros_like(img)
        return np.clip(img, 0, 255).astype(np.uint8)

    def __getitem__(self, index):
        if not self.init_seed:
            info = data.get_worker_info()
            if info is not None:
                torch.manual_seed(info.id)
                np.random.seed(info.id)
                random.seed(info.id)
                self.init_seed = True

        index = index % len(self.pairs)
        img1 = self._normalize(self._get_frame(self.pairs[index][0]))
        img2 = self._normalize(self._get_frame(self.pairs[index][1]))

        # grayscale -> 3ch
        if img1.ndim == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))

        h, w = self.target_size
        img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.zeros(2, h, w, dtype=torch.float32)
        valid = torch.ones(h, w, dtype=torch.float32)
        return img1, img2, flow, valid

    def __len__(self):
        return len(self.pairs)


def build_dataloaders(roots, pattern='envelope_flow*.mat',
                      target_size=(256, 256), val_ratio=0.1,
                      log_compress=False, cache_max=0,
                      batch_size=6, num_workers=8, persistent_workers=True):
    """Build train and (optionally) val DataLoaders for .mat envelope data."""
    train_ds = EnvelopeMatDataset(
        roots, pattern, target_size, split='train',
        val_ratio=val_ratio, log_compress=log_compress, cache_max=cache_max)

    loader_kw = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers,
                     drop_last=True, pin_memory=True)
    if persistent_workers and num_workers > 0:
        loader_kw['persistent_workers'] = True
    train_loader = data.DataLoader(train_ds, **loader_kw)

    val_loader = None
    if val_ratio > 0:
        val_ds = EnvelopeMatDataset(
            roots, pattern, target_size, split='val',
            val_ratio=val_ratio, log_compress=log_compress, cache_max=cache_max,
            verbose=False)
        val_loader = data.DataLoader(
            val_ds, batch_size=min(batch_size, 4), shuffle=False,
            num_workers=0, drop_last=False, pin_memory=True)

    print('Training with %d image pairs' % len(train_ds))
    if val_loader is not None:
        print('Validation with %d image pairs' % len(val_loader.dataset))
    return train_loader, val_loader
