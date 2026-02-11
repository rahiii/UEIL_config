import sys, argparse
from pathlib import Path

import yaml, torch, numpy as np, scipy.io as sio
from framework.adapter import load_adapter
from framework.data import build_dataloader


def pick_device(cfg):
    dev = str(cfg.get("runtime", {}).get("device", "auto")).lower()
    if dev == "cpu":   return torch.device("cpu")
    if dev == "mps":   return torch.device("mps")
    if dev == "cuda":  return torch.device("cuda")
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_batches", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists(): sys.exit(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.setdefault("runtime", {})["mode"] = "inference"

    model_name = cfg["model"]["name"]
    device = pick_device(cfg)

    inf = cfg.get("output", {}).get("inference", {})
    outdir = Path(inf.get("outdir", "outputs/inference"))
    if not outdir.is_absolute(): outdir = (repo_root / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    fmt = str(inf.get("format", "per_pair")).lower()
    filename = str(inf.get("filename", "disp_all.mat"))
    save_vis = bool(inf.get("save_vis", False))
    skip = bool(inf.get("skip_existing", False))

    print(f"model: {model_name} | device: {device} | outdir: {outdir}")

    adapter = load_adapter(cfg, repo_root)
    model = adapter.build(cfg).to(device)
    model.eval()

    loader = build_dataloader(cfg, repo_root=repo_root)
    print(f"pairs: {len(loader.dataset)}, batch_size: {loader.batch_size}")

    all_dx, all_dy, all_names = [], [], []
    vis_dir = outdir / "vis" if save_vis else None
    if vis_dir: vis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if args.max_batches and step >= args.max_batches: break

            out = adapter.run(model, batch, cfg)
            dx, dy = out["disp_x"].cpu(), out["disp_y"].cpu()
            meta = batch["meta"]

            for b in range(dx.shape[0]):
                p1, p2 = Path(meta["p1"][b]).stem, Path(meta["p2"][b]).stem

                if fmt == "per_pair":
                    fp = outdir / f"{p1}__{p2}_flow.mat"
                    if not (skip and fp.exists()):
                        sio.savemat(str(fp), {"disp_x": dx[b].numpy(), "disp_y": dy[b].numpy()})
                else:
                    all_dx.append(dx[b].numpy())
                    all_dy.append(dy[b].numpy())
                    all_names.append(f"{p1}__{p2}")

                if save_vis and "flow_vis_rgb" in out:
                    vp = vis_dir / f"{p1}__{p2}.png"
                    if not (skip and vp.exists()):
                        from PIL import Image
                        Image.fromarray(out["flow_vis_rgb"][b].permute(1,2,0).numpy(), "RGB").save(vp)

            if step % 10 == 0: print(f"batch {step}")

    if fmt == "single_mat" and all_dx:
        p = outdir / filename
        if not (skip and p.exists()):
            sio.savemat(str(p), {"disp_x": np.stack(all_dx), "disp_y": np.stack(all_dy), "names": np.array(all_names, dtype=object)})
            print(f"saved {p} ({len(all_names)} pairs)")
    elif fmt == "separate_xy" and all_dx:
        sio.savemat(str(outdir/"disp_x.mat"), {"disp_x": np.stack(all_dx)})
        sio.savemat(str(outdir/"disp_y.mat"), {"disp_y": np.stack(all_dy)})

    print("done.")

if __name__ == "__main__":
    main()
