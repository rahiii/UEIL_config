import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

from framework.loss import warp


def flow_to_rgb(flow):
    """Convert flow (2, H, W) tensor to an RGB numpy image (magnitude + direction)."""
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]
    mag = np.sqrt(dx ** 2 + dy ** 2)
    ang = np.arctan2(dy, dx)
    h = (ang + np.pi) / (2 * np.pi)
    s = np.ones_like(mag)
    v = np.clip(mag / (np.percentile(mag, 95) + 1e-6), 0, 1)
    rgb = mcolors.hsv_to_rgb(np.stack([h, s, v], axis=-1))
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def save_flow_figure(model, image1, image2, step, save_dir, forward_fn=None, args=None):
    """Save a 2x2 figure: frame1, frame2, flow RGB, warped frame2."""
    model.eval()
    with torch.no_grad():
        if forward_fn is not None:
            flow_preds = forward_fn(model, image1, image2, args)
        else:
            flow_preds = model(image1, image2, iters=12)
        flow = flow_preds[-1]
        warped = warp(image2 / 255.0, flow) * 255.0
    model.train()

    i = 0
    img1 = image1[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    img2 = image2[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    img2w = warped[i].permute(1, 2, 0).clamp(0, 255).cpu().numpy().astype(np.uint8)
    frgb = flow_to_rgb(flow[i])

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, im, title in zip(
        [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
        [img1, img2, frgb, img2w],
        ['Frame 1', 'Frame 2', 'Predicted flow', 'Frame 2 warped to 1'],
    ):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis('off')
    plt.suptitle('Step %d' % step)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'step_%07d.png' % step), dpi=120, bbox_inches='tight')
    plt.close()
