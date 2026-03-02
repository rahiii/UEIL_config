import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Generic optical-flow helpers (model-agnostic)
# ---------------------------------------------------------------------------

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device),
                            torch.arange(wd, device=device), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def bilinear_sampler(img, coords):
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    return F.grid_sample(img, grid, align_corners=True)


def warp(image, flow):
    """Warp image2 toward image1 using the predicted flow."""
    N, _, H, W = flow.shape
    coords = coords_grid(N, H, W, flow.device) + flow
    return bilinear_sampler(image, coords.permute(0, 2, 3, 1))


# ---------------------------------------------------------------------------
# Loss functions  — all share signature (image1, image2, flow_preds, **kw)
#                   and return (loss_tensor, metrics_dict)
# ---------------------------------------------------------------------------

def _smoothness(flow):
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def _ncc(img1, img2, win=9):
    """Local Normalized Cross-Correlation (single-channel)."""
    I = img1[:, :1] / 255.0
    J = img2[:, :1] / 255.0
    pad = win // 2
    filt = torch.ones(1, 1, win, win, device=I.device, dtype=I.dtype) / (win * win)
    I_mean = F.conv2d(I, filt, padding=pad)
    J_mean = F.conv2d(J, filt, padding=pad)
    cross = F.conv2d(I * J, filt, padding=pad) - I_mean * J_mean
    I_var = F.conv2d(I * I, filt, padding=pad) - I_mean * I_mean
    J_var = F.conv2d(J * J, filt, padding=pad) - J_mean * J_mean
    ncc = cross / (torch.sqrt(I_var * J_var + 1e-6) + 1e-6)
    return 1 - ncc.mean()


def ncc_loss(image1, image2, flow_preds, smooth_weight=0.001, mag_weight=0.01):
    """NCC photometric + flow smoothness + flow magnitude."""
    flow = flow_preds[-1]
    warped = warp(image2 / 255.0, flow) * 255.0
    photo = _ncc(image1, warped)
    smooth = _smoothness(flow)
    mag = flow.abs().mean()
    loss = photo + smooth_weight * smooth + mag_weight * mag
    metrics = {'ncc': photo.item(), 'smoothness': smooth.item(),
               'magnitude': mag.item(), 'total': loss.item()}
    return loss, metrics


def photometric_loss(image1, image2, flow_preds, smooth_weight=0.001, mag_weight=0.01):
    """Charbonnier photometric + flow smoothness + flow magnitude."""
    flow = flow_preds[-1]
    warped = warp(image2 / 255.0, flow) * 255.0
    diff = (image1 / 255.0 - warped / 255.0).abs()
    photo = ((diff ** 2 + 1e-6) ** 0.5).mean()
    smooth = _smoothness(flow)
    mag = flow.abs().mean()
    loss = photo + smooth_weight * smooth + mag_weight * mag
    metrics = {'photometric': photo.item(), 'smoothness': smooth.item(),
               'magnitude': mag.item(), 'total': loss.item()}
    return loss, metrics


# ---------------------------------------------------------------------------
# Registry — add new losses here, then use them from CLI with --loss <name>
# ---------------------------------------------------------------------------

LOSS_REGISTRY = {
    'ncc': ncc_loss,
    'photometric': photometric_loss,
}


def get_loss_fn(name):
    if name not in LOSS_REGISTRY:
        raise ValueError("Unknown loss '%s'. Available: %s" % (name, list(LOSS_REGISTRY.keys())))
    return LOSS_REGISTRY[name]
