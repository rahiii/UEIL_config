"""
General-purpose unsupervised training script.
Uses models from models/ and custom datasets + losses from framework/.

Example (RAFT on in-vivo .mat data with NCC loss):
    python train.py --model raft --loss ncc --data_root path/to/mat_folder \
        --num_steps 50000 --batch_size 12 --lr 2e-4 --image_size 256 256
"""
from __future__ import print_function, division

import sys
import os
import shutil
import argparse
import importlib.util
import time

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from framework.datasets import build_dataloaders
from framework.loss import get_loss_fn
from framework.visualization import save_flow_figure

try:
    from torch.cuda.amp import GradScaler
except Exception:
    class GradScaler:
        def __init__(self, **kw):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

from torch.utils.tensorboard import SummaryWriter


SUM_FREQ = 100


# ---------------------------------------------------------------------------
# Model factory — dynamically loads models/<name>/adapter.py
# ---------------------------------------------------------------------------

def _freeze_bn(model):
    """Safely call freeze_bn() whether or not the model is wrapped in DataParallel."""
    inner = model.module if hasattr(model, 'module') else model
    if hasattr(inner, 'freeze_bn'):
        inner.freeze_bn()


def build_model(args):
    """Load the adapter for args.model and return (model, forward_fn).

    Each model needs a models/<name>/adapter.py with two functions:
        build(args)                      -> nn.Module (wrapped in DataParallel)
        forward(model, img1, img2, args) -> list of flow preds [(B,2,H,W), ...]
    """
    adapter_path = os.path.join(PROJECT_ROOT, 'models', args.model, 'adapter.py')
    if not os.path.isfile(adapter_path):
        raise FileNotFoundError(
            "No adapter found at %s\n"
            "Each model needs an adapter.py — see README." % adapter_path
        )

    spec = importlib.util.spec_from_file_location(args.model + '_adapter', adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    model = adapter.build(args)
    forward_fn = adapter.forward

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()
    _freeze_bn(model)

    return model, forward_fn


# ---------------------------------------------------------------------------
# Logger (tensorboard)
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, scheduler, log_dir):
        self.scheduler = scheduler
        self.log_dir = log_dir
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _get_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        return self.writer

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)
        print(training_str + metrics_str)
        w = self._get_writer()
        for k in self.running_loss:
            w.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1
        for k in metrics:
            if k not in self.running_loss:
                self.running_loss[k] = 0.0
            self.running_loss[k] += metrics[k]
        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        w = self._get_writer()
        for k in results:
            w.add_scalar(k, results[k], self.total_steps)

    def close(self):
        if self.writer is not None:
            self.writer.close()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def setup_run_dir(args):
    """Create runs/<model>_<name>_<timestamp>/ with checkpoints/ and vis/ inside."""
    timestamp = time.strftime("%b_%d_%H_%M_%S")
    run_name = "%s_%s_%s" % (args.model, args.name, timestamp)
    run_dir = os.path.join(PROJECT_ROOT, 'runs', run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    vis_dir = os.path.join(run_dir, 'vis')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    cfg_snapshot = os.path.join(run_dir, 'config.yaml')
    with open(cfg_snapshot, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)
    print("Config saved to: %s" % cfg_snapshot)

    args.run_dir = run_dir
    args.ckpt_dir = ckpt_dir
    args.vis_dir = vis_dir
    return args


def train(args):
    args = setup_run_dir(args)
    start_time = time.time()
    print("Run started at %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Run directory: %s" % args.run_dir)

    model, forward_fn = build_model(args)
    print("Parameter Count: %d" % sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Dataset
    data_roots = args.data_root or ['datasets/invivo']
    train_loader, val_loader = build_dataloaders(
        roots=data_roots,
        pattern=args.mat_pattern,
        target_size=args.image_size,
        val_ratio=args.val_ratio,
        log_compress=args.log_compress,
        cache_max=args.cache_max,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=not args.no_persistent_workers,
    )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(scheduler, log_dir=args.run_dir)
    loss_fn = get_loss_fn(args.loss)

    # NCC relies on speckle correlation — adding noise destroys it
    use_noise = args.add_noise and args.loss != 'ncc'

    VAL_FREQ = int(getattr(args, 'checkpoint_freq', 5000))
    total_steps = 0
    should_keep_training = True

    while should_keep_training:
        for data_blob in train_loader:
            optimizer.zero_grad()
            image1, image2, flow_gt, valid = [x.cuda() for x in data_blob]

            if use_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn_like(image1)).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn_like(image2)).clamp(0.0, 255.0)

            flow_preds = forward_fn(model, image1, image2, args)
            loss, metrics = loss_fn(image1, image2, flow_preds,
                                    smooth_weight=args.smooth_weight,
                                    mag_weight=args.mag_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            # Visualization
            if args.vis_freq > 0 and (total_steps + 1) % int(args.vis_freq) == 0:
                vis_path = os.path.join(args.vis_dir, 'step_%07d.png' % (total_steps + 1))
                save_flow_figure(model, image1, image2, total_steps + 1,
                                 args.vis_dir, forward_fn=forward_fn, args=args)
                print("  Saved visualization: %s" % vis_path)

            # Checkpoint + validation
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                elapsed = (time.time() - start_time) / 60.0
                print("  elapsed: %.1f min" % elapsed)

                ckpt_path = os.path.join(args.ckpt_dir, '%d_%s.pth' % (total_steps + 1, args.name))
                torch.save(model.state_dict(), ckpt_path)
                print("  Saved checkpoint: %s" % ckpt_path)

                if val_loader is not None:
                    model.eval()
                    val_total, val_n = 0.0, 0
                    with torch.no_grad():
                        for vi, vblob in enumerate(val_loader):
                            if vi >= 20:
                                break
                            v1, v2, _, _ = [x.cuda() for x in vblob]
                            vpreds = forward_fn(model, v1, v2, args)
                            vloss, _ = loss_fn(v1, v2, vpreds,
                                               smooth_weight=args.smooth_weight,
                                               mag_weight=args.mag_weight)
                            val_total += vloss.item()
                            val_n += 1
                    model.train()
                    _freeze_bn(model)
                    if val_n > 0:
                        avg = val_total / val_n
                        print("  val_loss=%.4f (step %d)" % (avg, total_steps + 1))
                        logger.write_dict({'val/loss': avg})

                model.train()
                _freeze_bn(model)

            total_steps += 1
            if total_steps > args.num_steps:
                should_keep_training = False
                break

    elapsed = time.time() - start_time
    print("Total run time: %.1f s (%.1f min)" % (elapsed, elapsed / 60.0))
    logger.close()

    final_path = os.path.join(args.ckpt_dir, '%s_final.pth' % args.name)
    torch.save(model.state_dict(), final_path)
    print("Final checkpoint: %s" % final_path)
    return final_path


# ---------------------------------------------------------------------------
# CLI — loads config.yaml first, then CLI flags override any value
# ---------------------------------------------------------------------------

def load_config(config_path, cli_args):
    """Merge YAML config with CLI overrides into an argparse.Namespace."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # CLI overrides: any flag explicitly passed on the command line wins
    for key, val in vars(cli_args).items():
        if key == 'config':
            continue
        if val is not None:
            cfg[key] = val

    # Convert booleans that YAML already parsed correctly
    return argparse.Namespace(**cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml', help='path to YAML config')

    # Every field can also be overridden from CLI
    parser.add_argument('--name', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--loss', default=None)
    parser.add_argument('--restore_ckpt', default=None)
    parser.add_argument('--gpus', type=int, nargs='+', default=None)
    parser.add_argument('--mixed_precision', action='store_true', default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_steps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--image_size', type=int, nargs='+', default=None)
    parser.add_argument('--clip', type=float, default=None)
    parser.add_argument('--wdecay', type=float, default=None)
    parser.add_argument('--epsilon', type=float, default=None)
    parser.add_argument('--add_noise', action='store_true', default=None)
    parser.add_argument('--smooth_weight', type=float, default=None)
    parser.add_argument('--mag_weight', type=float, default=None)
    parser.add_argument('--data_root', type=str, nargs='+', default=None)
    parser.add_argument('--mat_pattern', type=str, default=None)
    parser.add_argument('--val_ratio', type=float, default=None)
    parser.add_argument('--log_compress', action='store_true', default=None)
    parser.add_argument('--cache_max', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--no_persistent_workers', action='store_true', default=None)
    parser.add_argument('--checkpoint_freq', type=int, default=None)
    parser.add_argument('--vis_freq', type=int, default=None)
    parser.add_argument('--small', action='store_true', default=None)
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--alternate_corr', action='store_true', default=None)
    parser.add_argument('--vxm_features', type=int, nargs='+', default=None)
    parser.add_argument('--vxm_int_steps', type=int, default=None)

    cli_args = parser.parse_args()
    args = load_config(cli_args.config, cli_args)

    torch.manual_seed(1234)
    np.random.seed(1234)

    train(args)
