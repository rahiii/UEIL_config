from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _warp_img_with_flow(
    img: torch.Tensor,
    flow: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Warp img2 into img1's coordinate frame using optical flow.

    Args:
      img:  [B, C, H, W]
      flow: [B, 2, H, W]  (flow in x/y pixel units)

    Returns:
      {
        "warped": [B, C, H, W],
        "valid":  [B, H, W]  (1 where sampling is inside image bounds)
      }
    """
    if img.shape[0] != flow.shape[0] or img.shape[2:] != flow.shape[2:]:
        raise ValueError(
            f"Image and flow must have same batch/height/width. "
            f"Got img={tuple(img.shape)}, flow={tuple(flow.shape)}"
        )

    B, C, H, W = img.shape

    # base grid of (x, y) pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(H, device=img.device),
        torch.arange(W, device=img.device),
        indexing="ij",
    )
    x = x.float().unsqueeze(0).expand(B, -1, -1)  # [B,H,W]
    y = y.float().unsqueeze(0).expand(B, -1, -1)

    fx = flow[:, 0]  # [B,H,W]
    fy = flow[:, 1]

    x_new = x + fx
    y_new = y + fy

    # valid mask: points that land inside image
    valid = (
        (x_new >= 0.0)
        & (x_new <= (W - 1))
        & (y_new >= 0.0)
        & (y_new <= (H - 1))
    )

    # normalize to [-1, 1] for grid_sample
    x_norm = 2.0 * (x_new / max(W - 1, 1)) - 1.0
    y_norm = 2.0 * (y_new / max(H - 1, 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)  # [B,H,W,2]

    warped = F.grid_sample(img, grid, align_corners=True)
    return {"warped": warped, "valid": valid.float()}


def _flow_smooth_l1(flow: torch.Tensor) -> torch.Tensor:
    """First-order L1 smoothness regularizer on flow [B,2,H,W]."""
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return 0.5 * (dx.abs().mean() + dy.abs().mean())


def _mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Mean squared error over all pixels/channels."""
    return F.mse_loss(img1.float(), img2.float())


def _mae(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Mean absolute error (L1) over all pixels/channels."""
    return F.l1_loss(img1.float(), img2.float())


def _ncc(img1: torch.Tensor, img2: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Simple global normalized cross-correlation (NCC) similarity.
    Higher is better; we normally turn it into a loss as (1 - NCC).
    """
    x = img1.float()
    y = img2.float()

    x = x - x.mean()
    y = y - y.mean()

    num = (x * y).mean()
    den = torch.sqrt((x * x).mean() * (y * y).mean()) + eps
    return num / den


# ---------------------------------------------------------------------------
# High-level loss functions (used by config)
# ---------------------------------------------------------------------------

def loss_mse(img1, img2, **_) -> Dict[str, torch.Tensor]:
    """Plain MSE between two images (ignores flow)."""
    val = _mse(img1, img2)
    return {"total": val, "mse": val}


def loss_l1(img1, img2, **_) -> Dict[str, torch.Tensor]:
    """Plain L1/MAE between two images (ignores flow)."""
    val = _mae(img1, img2)
    return {"total": val, "l1": val}


def loss_photometric_smooth(
    img1,
    img2,
    flow,
    w_photo: float,
    w_smooth: float,
    **_,
) -> Dict[str, torch.Tensor]:
    """
    Unsupervised photometric + smoothness objective:
      - photometric term between img1 and warped img2
      - first-order smoothness on flow
    """
    img1 = img1.float()
    img2 = img2.float()
    flow = flow.float()

    warp_out = _warp_img_with_flow(img2, flow)
    img2_w = warp_out["warped"]        # [B,C,H,W]
    valid = warp_out["valid"]          # [B,H,W]

    diff = (img1 - img2_w).abs()       # [B,C,H,W]
    valid_exp = valid.unsqueeze(1)     # [B,1,H,W]
    diff = diff * valid_exp

    denom = valid_exp.sum().clamp(min=1.0)
    loss_photo = diff.sum() / denom

    loss_smooth = _flow_smooth_l1(flow)

    total = w_photo * loss_photo + w_smooth * loss_smooth
    return {
        "total": total,
        "photo": loss_photo,
        "smooth": loss_smooth,
    }


def loss_ncc_smooth(
    img1,
    img2,
    flow,
    w_ncc: float,
    w_smooth: float,
    **_,
) -> Dict[str, torch.Tensor]:
    """
    NCC + smoothness objective:
      - NCC similarity between img1 and img2
      - smoothness on flow
    Loss = w_smooth * smooth - w_ncc * NCC   (maximize NCC, minimize smoothness)
    """
    ncc = _ncc(img1, img2)
    loss_smooth = _flow_smooth_l1(flow)
    total = w_smooth * loss_smooth - w_ncc * ncc
    return {
        "total": total,
        "ncc": ncc,
        "smooth": loss_smooth,
    }


# ---------------------------------------------------------------------------
# Dispatcher used by train.py
# ---------------------------------------------------------------------------

def compute_loss(
    loss_cfg: Dict[str, Any],
    *,
    img1,
    img2,
    flow,
    model=None,
    batch=None,
) -> Dict[str, Any]:
    """
    Central loss dispatcher for the framework.

    Config examples:

      # Unsupervised photometric + smoothness
      training:
        loss:
          type: photometric_smooth
          w_photo: 1.0
          w_smooth: 0.05

      # Simple MSE between img1 and img2 (ignores flow)
      training:
        loss:
          type: mse

      # NCC + smoothness
      training:
        loss:
          type: ncc_smooth
          w_ncc: 1.0
          w_smooth: 0.05
    """
    ltype = str(loss_cfg.get("type", "photometric_smooth")).lower()

    if ltype in ("none", "off", "disabled"):
        zero = flow.new_tensor(0.0)
        return {"total": zero}

    if ltype in ("mse", "l2"):
        return loss_mse(img1=img1, img2=img2)

    if ltype in ("l1", "mae"):
        return loss_l1(img1=img1, img2=img2)

    if ltype in ("photometric_smooth", "photo_smooth"):
        return loss_photometric_smooth(
            img1=img1,
            img2=img2,
            flow=flow,
            w_photo=float(loss_cfg.get("w_photo", 1.0)),
            w_smooth=float(loss_cfg.get("w_smooth", 0.05)),
        )

    if ltype in ("ncc_smooth", "vmorph"):
        return loss_ncc_smooth(
            img1=img1,
            img2=img2,
            flow=flow,
            w_ncc=float(loss_cfg.get("w_ncc", 1.0)),
            w_smooth=float(loss_cfg.get("w_smooth", 0.05)),
        )

    raise ValueError(f"Unknown loss.type: {ltype}")

