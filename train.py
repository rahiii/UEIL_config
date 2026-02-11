import os, sys, argparse, json
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import yaml, torch
from framework.adapter import load_adapter
from framework.data import build_train_val_loaders
from framework.loss import compute_loss


def pick_device(cfg):
    dev = str(cfg.get("runtime", {}).get("device", "auto")).lower()
    if dev == "cpu":   return torch.device("cpu")
    if dev == "mps":   return torch.device("mps")
    if dev == "cuda":  return torch.device("cuda")
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def make_run_dir(repo_root, cfg):
    run_dir = repo_root / str(cfg.get("output", {}).get("run_dir", "runs/run"))
    if run_dir.exists():
        base, i = run_dir, 2
        while run_dir.exists():
            run_dir = base.parent / f"{base.name}_{i}"; i += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists(): sys.exit(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())

    model_name = cfg["model"]["name"]
    device = pick_device(cfg)
    run_dir = make_run_dir(repo_root, cfg)
    print(f"model: {model_name} | device: {device} | run_dir: {run_dir}")

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    metrics_path = run_dir / "metrics.jsonl"
    vis_dir = run_dir / "vis"; vis_dir.mkdir(exist_ok=True)

    adapter = load_adapter(cfg, repo_root)
    model = adapter.build(cfg).to(device)
    model.train()

    train_loader, val_loader = build_train_val_loaders(cfg, repo_root=repo_root)
    print(f"train pairs: {len(train_loader.dataset)}, batch_size: {train_loader.batch_size}")
    if val_loader is not None:
        print(f"val pairs:   {len(val_loader.dataset)}")
    else:
        print("validation: disabled (validation_split <= 0)")

    tcfg = cfg.get("training", {})
    steps = int(tcfg.get("num_steps", 100))
    lr = float(tcfg.get("lr", 1e-5))
    loss_cfg = tcfg.get("loss", {}) or {}
    log_freq = int(cfg.get("runtime", {}).get("log_freq", 5))
    val_freq = int(tcfg.get("val_freq", 0))
    out_cfg = cfg.get("output", {})

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    it = iter(train_loader)

    for step in range(steps):
        # ── Training step ────────────────────────────────────────
        try: batch = next(it)
        except StopIteration: it = iter(train_loader); batch = next(it)

        out = adapter.run(model, batch, cfg)
        losses = compute_loss(loss_cfg, img1=batch["img1"].to(device), img2=batch["img2"].to(device), flow=out["flow"])

        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        opt.step()

        rec = {"step": step, "total": float(losses["total"].detach().cpu())}
        for k in ("photo", "smooth", "ncc", "mse", "l1"):
            if k in losses: rec[k] = float(losses[k].detach().cpu())
        with metrics_path.open("a") as f: f.write(json.dumps(rec) + "\n")

        if log_freq and step % log_freq == 0:
            print(f"step {step} | " + " ".join(f"{k}={v:.6f}" for k,v in rec.items() if k != "step"))

        # ── Validation ───────────────────────────────────────────
        if val_loader is not None and val_freq > 0 and step % val_freq == 0:
            model.eval()
            val_totals = {k: 0.0 for k in ["total", "photo", "smooth", "ncc", "mse", "l1"]}
            val_count = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_out = adapter.run(model, val_batch, cfg)
                    val_losses = compute_loss(
                        loss_cfg,
                        img1=val_batch["img1"].to(device),
                        img2=val_batch["img2"].to(device),
                        flow=val_out["flow"],
                    )
                    for k in val_totals:
                        if k in val_losses:
                            val_totals[k] += float(val_losses[k].detach().cpu())
                    val_count += 1

            if val_count > 0:
                val_rec = {"step": step}
                for k in val_totals:
                    avg = val_totals[k] / val_count
                    if avg != 0.0:
                        val_rec[f"val_{k}"] = avg
                with metrics_path.open("a") as f: f.write(json.dumps(val_rec) + "\n")
                print(f"step {step} | " + " ".join(f"{k}={v:.6f}" for k,v in val_rec.items() if k != "step"))

            model.train()

        # ── Checkpoints & vis ────────────────────────────────────
        sf = int(out_cfg.get("save_checkpoint_freq", 50))
        if sf and step % sf == 0:
            torch.save({"step": step, "state_dict": model.state_dict(), "cfg": cfg}, run_dir / f"ckpt_{step:06d}.pt")

        vf = int(out_cfg.get("save_vis_freq", 50))
        if vf and step % vf == 0 and "flow_vis_rgb" in out:
            from PIL import Image
            arr = out["flow_vis_rgb"][0].permute(1,2,0).contiguous().numpy()
            Image.fromarray(arr, "RGB").save(vis_dir / f"step{step:06d}.png")

    print(f"done → {run_dir}")

if __name__ == "__main__":
    main()
