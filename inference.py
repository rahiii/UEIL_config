# inference.py
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

import yaml
import torch
import numpy as np
import scipy.io as sio

from framework.registry import load_adapter
from framework.datasets import build_dataloader


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


def move_batch_to_device(batch, device: torch.device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def save_flow_mat(path: Path, disp_x: torch.Tensor, disp_y: torch.Tensor) -> None:
    sio.savemat(str(path), {"disp_x": disp_x.numpy(), "disp_y": disp_y.numpy()})


def save_rgb_png(path: Path, rgb_chw_u8: torch.Tensor) -> None:
    from PIL import Image
    arr = rgb_chw_u8.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(arr, mode="RGB").save(path)


def _stem_noext(p: str) -> str:
    return Path(p).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override_outdir", default="", help="Optional override output directory")
    ap.add_argument("--max_batches", type=int, default=0, help="0 = all")
    ap.add_argument("--skip_existing", action="store_true", help="Skip existing output files (overrides config)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print("Config not found:", cfg_path)
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text())

    # Model name must be provided explicitly in the config (inference is model-agnostic).
    model_name = str(cfg.get("model", "")).strip().lower()
    if not model_name:
        print("Error: config is missing required field 'model'.")
        sys.exit(1)
    device = pick_device(cfg)

    # Read inference config from output.inference
    output_cfg = cfg.get("output", {})
    inf_cfg = output_cfg.get("inference", {})
    
    outdir = Path(args.override_outdir or inf_cfg.get("outdir", "outputs/inference")).expanduser()
    outdir = (repo_root / outdir).resolve() if not outdir.is_absolute() else outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    
    format_mode = str(inf_cfg.get("format", "per_pair")).lower()
    filename = str(inf_cfg.get("filename", "disp_all.mat"))
    save_vis = bool(inf_cfg.get("save_vis", False))
    # Command-line argument overrides config
    skip_existing = args.skip_existing if args.skip_existing else bool(inf_cfg.get("skip_existing", False))

    print("repo_root:", repo_root)
    print("config:", cfg_path)
    print("model:", model_name)
    print("device:", device)
    print("outdir:", outdir)
    print("format:", format_mode)
    print("save_vis:", save_vis)
    print("skip_existing:", skip_existing)

    # Force deterministic order for inference
    # (build_dataloader should read this and set shuffle accordingly)
    cfg = dict(cfg)
    cfg.setdefault("runtime", {})
    cfg["runtime"] = dict(cfg["runtime"])
    cfg["runtime"]["mode"] = "inference"  # lets datasets.py choose shuffle=False

    adapter = load_adapter(model_name)
    model = adapter.build(cfg, repo_root=repo_root).to(device)
    model.eval()

    loader = build_dataloader(cfg, repo_root=repo_root)
    print("num pairs:", len(loader.dataset))
    print("batch size:", loader.batch_size)

    # For single_mat and separate_xy formats, accumulate all results
    all_disp_x: List[np.ndarray] = []
    all_disp_y: List[np.ndarray] = []
    all_names: List[str] = []

    vis_dir = outdir / "vis" if save_vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if args.max_batches and step >= args.max_batches:
                break

            batch = move_batch_to_device(batch, device)
            batch["_test_mode"] = True

            out = adapter.forward(model, batch, cfg)

            disp_x = out["disp_x"].detach().cpu()  # [B,H,W]
            disp_y = out["disp_y"].detach().cpu()

            meta = batch["meta"]
            p1_list = meta["p1"]
            p2_list = meta["p2"]

            for b in range(disp_x.shape[0]):
                p1 = _stem_noext(p1_list[b])
                p2 = _stem_noext(p2_list[b])
                
                if format_mode == "per_pair":
                    fname = outdir / f"{p1}__{p2}_flow.mat"
                    
                    if skip_existing and fname.exists():
                        print(f"Skipping existing: {fname.name}")
                        continue
                    
                    save_flow_mat(fname, disp_x[b], disp_y[b])
                    
                elif format_mode in ["single_mat", "separate_xy"]:
                    # Accumulate for single_mat or separate_xy format
                    all_disp_x.append(disp_x[b].numpy())
                    all_disp_y.append(disp_y[b].numpy())
                    all_names.append(f"{p1}__{p2}")
                
                # Save visualization if requested
                if save_vis and "flow_vis_rgb" in out:
                    vis_fname = vis_dir / f"{p1}__{p2}_flow.png"
                    if not (skip_existing and vis_fname.exists()):
                        save_rgb_png(vis_fname, out["flow_vis_rgb"][b])

            if step % 10 == 0:
                print(f"inference batch {step}")

    # Save single_mat if format is single_mat
    if format_mode == "single_mat":
        single_mat_path = outdir / filename
        if skip_existing and single_mat_path.exists():
            print(f"Skipping existing: {single_mat_path.name}")
        else:
            # Stack all displacements into arrays
            # Shape: [N, H, W] for each
            disp_x_stack = np.stack(all_disp_x, axis=0)  # [N, H, W]
            disp_y_stack = np.stack(all_disp_y, axis=0)   # [N, H, W]
            
            save_dict = {
                "disp_x": disp_x_stack,
                "disp_y": disp_y_stack,
                "names": np.array(all_names, dtype=object),
            }
            sio.savemat(str(single_mat_path), save_dict)
            print(f"Saved single_mat: {single_mat_path} with {len(all_names)} pairs")
    
    # Save separate_xy if format is separate_xy
    elif format_mode == "separate_xy":
        disp_x_path = outdir / "disp_x.mat"
        disp_y_path = outdir / "disp_y.mat"
        
        if skip_existing and disp_x_path.exists() and disp_y_path.exists():
            print(f"Skipping existing: {disp_x_path.name} and {disp_y_path.name}")
        else:
            # Stack all displacements into arrays
            # Shape: [N, H, W] for each
            disp_x_stack = np.stack(all_disp_x, axis=0)  # [N, H, W]
            disp_y_stack = np.stack(all_disp_y, axis=0)   # [N, H, W]
            
            sio.savemat(str(disp_x_path), {"disp_x": disp_x_stack})
            sio.savemat(str(disp_y_path), {"disp_y": disp_y_stack})
            print(f"Saved separate_xy: {disp_x_path} and {disp_y_path} with {len(all_names)} pairs")

    print("done.")


if __name__ == "__main__":
    main()
