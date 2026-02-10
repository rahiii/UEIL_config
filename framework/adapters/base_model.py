"""
Base adapter template for a new model.

How to use this file:
1) Copy it to: framework/adapters/<your_model_name>.py
2) Rename `BaseModelAdapter` (or just keep `Adapter`) and update the logic.
3) Make sure your config's `model:` matches the adapter file name (e.g. "base_model").
4) Implement `build`, `forward`, and (optionally) `compute_loss`.

See `framework/adapters/raft.py` and `framework/adapters/voxelmorph.py`
for fully working examples.
"""

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import torch


class Adapter:
    """
    Minimal template for a framework adapter.

    The training and inference scripts expect the adapter to provide:
      - build(cfg, repo_root) -> torch.nn.Module
      - forward(model, batch, cfg) -> dict with:
          - "disp_x": displacement field in x direction, shape [B, H, W]
          - "disp_y": displacement field in y direction, shape [B, H, W]
          - optionally "flow_vis_rgb": visualization tensor [B, 3, H, W] (uint8 or float)
      - compute_loss(img1, img2, flow, loss_cfg, model) -> dict with:
          - at least "total": scalar loss tensor
          - optionally sub-losses (e.g. "image", "smooth", etc.)
    """

    def __init__(self):
        print("Initializing base_model adapter template")

    def build(self, cfg: Dict[str, Any], repo_root: Path) -> torch.nn.Module:
        """
        Build and return the model.

        Typical steps:
          - read settings from cfg["models"]["base_model"]["model_settings"]
          - import your model code from models/base_model/ or an installed package
          - construct and return a torch.nn.Module
        """
        # Example skeleton (replace with your real model):
        #
        # from models.base_model.model import MyModel
        # settings = cfg.get("models", {}).get("base_model", {}).get("model_settings", {}) or {}
        # model = MyModel(**settings)
        # return model
        #
        raise NotImplementedError("Implement build() for your model in this adapter.")

    def forward(
        self, model: torch.nn.Module, batch: Dict[str, Any], cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a forward pass.

        Expected batch keys (see your dataset implementation for details):
          - "img1": first image tensor [B, C, H, W]
          - "img2": second image tensor [B, C, H, W]

        This method must return at least:
          - "disp_x": [B, H, W] displacement in x
          - "disp_y": [B, H, W] displacement in y

        Optionally it can return:
          - "flow_vis_rgb": [B, 3, H, W] visualization for saving PNGs
        """
        raise NotImplementedError(
            "Implement forward() to run your model and produce disp_x/disp_y."
        )

    def compute_loss(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        flow: torch.Tensor,
        loss_cfg: Dict[str, Any],
        model: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        The trainer expects:
          - a dict with at least a "total" key (scalar tensor)
          - optionally sub-losses like "image", "smooth", etc.

        Example pattern:
          losses = {}
          losses["image"] = image_loss(...)
          losses["smooth"] = smoothness_loss(flow, ...)
          losses["total"] = (
              w_image * losses["image"] +
              w_smooth * losses["smooth"]
          )
          return losses
        """
        raise NotImplementedError(
            "Implement compute_loss() for your model or remove training usage."
        )


