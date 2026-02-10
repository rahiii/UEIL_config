import os
import sys
import argparse
import json
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import yaml
import torch

from framework.registry import load_adapter
from framework.datasets import build_dataloader
from framework.loss import compute_loss


def pick_device(cfg) -> torch.device:
    dev = str(cfg.get("runtime", {}).get("device", "auto")).lower()
    if dev == "cpu":
        return torch.device("cpu")
    if dev == "mps":
        return torch.device("mps")
    if dev == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_run_dir(repo_root: Path, cfg: dict) -> Path:
    out = cfg.get("output", {})
    base = repo_root / str(out.get("path", "runs"))
    name = str(out.get("name", "run"))

    run_dir = base / name
    i = 2
    while run_dir.exists():
        run_dir = base / f"{name}_{i}"
        i += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_rgb_png(path: Path, rgb_chw_u8: torch.Tensor) -> None:
    from PIL import Image
    arr = rgb_chw_u8.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(arr, mode="RGB").save(path)


def save_flow_mat(path: Path, disp_x: torch.Tensor, disp_y: torch.Tensor) -> None:
    import scipy.io as sio
    sio.savemat(str(path), {"disp_x": disp_x.numpy(), "disp_y": disp_y.numpy()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg_path = Path(args.config).resolve()

    print("Repo root:", repo_root)
    print("config path:", cfg_path)

    if not cfg_path.exists():
        print("Config not found:", cfg_path)
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text())

    # Model name must be provided explicitly in the config (train.py is model-agnostic).
    model_name = str(cfg.get("model", "")).strip().lower()
    if not model_name:
        print("Error: config is missing required field 'model'.")
        sys.exit(1)
    print("model from config:", model_name)

    device = pick_device(cfg)
    print("device:", device)

    run_dir = make_run_dir(repo_root, cfg)
    print("run_dir:", run_dir)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    metrics_path = run_dir / "metrics.jsonl"

    vis_dir = run_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    adapter = load_adapter(model_name)

    # IMPORTANT: framework adapters can need repo_root
    model = adapter.build(cfg, repo_root=repo_root).to(device)
    model.train()

    loader = build_dataloader(cfg, repo_root=repo_root)
    print("num pairs:", len(loader.dataset))
    print("batch size:", loader.batch_size)

    train_cfg = cfg.get("training", {}).get(model_name, {})
    runtime = cfg.get("runtime", {})
    out_cfg = cfg.get("output", {})

    steps = int(train_cfg.get("num_steps", 100))
    lr = float(train_cfg.get("lr", 1e-5))

    loss_cfg = train_cfg.get("loss", {}) or {}

    log_freq = int(runtime.get("log_freq", 5))

    save_flow_freq = int(out_cfg.get("save_displacement_freq", 50))
    save_vis = bool(out_cfg.get("save_vis", True))
    save_vis_freq = int(out_cfg.get("save_vis_freq", 50))

    save_ckpt = bool(out_cfg.get("save_checkpoint", True))
    save_ckpt_freq = int(out_cfg.get("save_checkpoint_freq", 50))

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    it = iter(loader)
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch["_test_mode"] = False

        out = adapter.forward(model, batch, cfg)
        flow = out["flow_device"]

        img1 = batch["img1"].to(device)
        img2 = batch["img2"].to(device)

        # Central, config-driven loss selection
        losses = compute_loss(
            loss_cfg,
            img1=img1,
            img2=img2,
            flow=flow,
            model=model,
            batch=batch,
        )

        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        opt.step()

        # Build metrics record (handle different loss key names)
        rec = {
            "step": step,
            "total": float(losses["total"].detach().cpu()),
        }
        # Add loss components (photo/image, smooth)
        if "photo" in losses:
            rec["photo"] = float(losses["photo"].detach().cpu())
        if "image" in losses:
            rec["image"] = float(losses["image"].detach().cpu())
        if "smooth" in losses:
            rec["smooth"] = float(losses["smooth"].detach().cpu())
        with metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        if log_freq and (step % log_freq == 0):
            loss_str = f"total={rec['total']:.6f}"
            if "photo" in rec:
                loss_str += f" photo={rec['photo']:.6f}"
            if "image" in rec:
                loss_str += f" image={rec['image']:.6f}"
            if "smooth" in rec:
                loss_str += f" smooth={rec['smooth']:.6f}"
            print(f"step {step} | {loss_str}")

        if save_ckpt and save_ckpt_freq and (step % save_ckpt_freq == 0):
            torch.save(
                {"step": step, "state_dict": model.state_dict(), "opt_state": opt.state_dict(), "cfg": cfg},
                run_dir / f"ckpt_step{step:06d}.pt",
            )

        if save_flow_freq and (step % save_flow_freq == 0):
            disp_x = out["disp_x"][0]
            disp_y = out["disp_y"][0]
            save_flow_mat(run_dir / f"train_step{step:06d}_b0_flow.mat", disp_x, disp_y)

        if save_vis and save_vis_freq and (step % save_vis_freq == 0):
            vis = out["flow_vis_rgb"][0]
            save_rgb_png(vis_dir / f"train_step{step:06d}_b0.png", vis)

    print("done. outputs saved to:", run_dir)


if __name__ == "__main__":
    main()
