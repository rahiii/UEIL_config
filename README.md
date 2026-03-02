# UEIL

Unsupervised optical-flow training framework for ultrasound envelope data. You add optical-flow or registration models into `models/` and train them on `.mat` envelope sequences with config-driven runs and optional mixed-precision training. **This repo does not include any models** — you must clone or add a model (e.g. RAFT, VoxelMorph) first.

---

## Requirements

- **Python** 3.10+ (recommended for conda envs)
- **PyTorch** with CUDA (GPU required for training)
- **Conda** (used by `setup_model.ps1` to create per-model environments)
- **Framework deps:** `pyyaml`, `scipy`, `opencv-python`, `matplotlib`, `tensorboard`

---

## Environments and `setup_model.ps1`

Training uses **one conda environment per model**. The environment name is the same as the model folder name under `models/` (e.g. `raft`, `voxelmorph`). You create and populate that environment with **`setup_model.ps1`** (Windows PowerShell, run from the project root).

**What `setup_model.ps1` does:**

1. **Preflight** — Checks that `models/<ModelName>/` exists; exits with an error if the model folder is missing.
2. **GPU detection** — Runs `nvidia-smi`, infers compute capability, and picks a PyTorch CUDA build: `cu118` (Turing), `cu121` (Ampere), `cu124` (Ada/Hopper), or `cu128` (Blackwell / default). This avoids installing a PyTorch build that doesn’t support your GPU.
3. **Conda env** — Creates a conda environment named `<ModelName>` (e.g. `conda create -n raft python=3.10`). If it already exists, creation is skipped.
4. **PyTorch** — Installs `torch`, `torchvision`, `torchaudio` from the PyTorch index for the chosen CUDA version.
5. **Model dependencies** — Scans `models/<ModelName>/` for `requirements*.txt`, `setup.py`, `pyproject.toml`, or `environment*.yml` and installs dependencies (excluding PyTorch/torch). Can also parse README for `pip install` / `conda install` lines as fallback.
6. **Framework dependencies** — Installs `pyyaml`, `scipy`, `opencv-python`, `matplotlib`, `tensorboard`.
7. **GPU verification** — Runs a short script to confirm CUDA is available and a small GPU tensor op works. If it fails, the script tries reinstalling PyTorch with `cu128` and re-verifies.

**Usage:**

```powershell
.\setup_model.ps1 <model_name>
.\setup_model.ps1 <model_name> -Python 3.11
```

Example: after cloning RAFT into `models/raft/`, run `.\setup_model.ps1 raft`. Then activate the env with `conda activate raft` and run training from the project root (see “Adding a new model” and “Training” below).

**Important:** You must have the model in `models/<name>/` before running `setup_model.ps1`; the script will not create the model folder for you.

---

## Setup (first-time workflow)

1. **Add a model** — Clone or copy the model repo into `models/<name>/` (e.g. `models/raft/`, `models/voxelmorph/`). No models are included in this repository.

2. **Create the conda environment for that model** (Windows PowerShell, from project root):
   ```powershell
   .\setup_model.ps1 <model_name>
   # Optional: use a specific Python version
   .\setup_model.ps1 <model_name> -Python 3.11
   ```
   This creates the conda env named `<model_name>`, installs PyTorch (with the correct CUDA build), the model’s dependencies, and the framework dependencies. Activate with:
   ```powershell
   conda activate <model_name>
   ```

3. **Create adapter + config** (after the model is in place and the env exists, if the model doesn’t already have an adapter):
   ```bash
   python create_adapter.py --name <model_name>
   # Optional: also run setup_model.ps1 (e.g. if you haven’t run it yet)
   python create_adapter.py --name <model_name> --setup
   # Overwrite existing adapter and config
   python create_adapter.py --name <model_name> --force
   ```
   This generates `models/<name>/adapter.py` and `config/<name>.yaml`. Fix the import and constructor in `adapter.py` if needed. Then you can train using that config (see “Adding a new model” and “Training” below).

---

## Data format

- **Input:** Consecutive frame pairs from ultrasound envelope data stored as **`.mat` files**.
- **Layout:** One or more root directories; each should contain `.mat` files (e.g. `envelope_flow1.mat`, `envelope_flow2.mat`, …). Pairs are formed from consecutive files (never across folders).
- **Mat content:** Each file should contain a 2D array under a key such as `img`, `envelope`, `data`, `im`, `frame`, or `I` (or the first 2D array found is used). See `framework/datasets.py` (`_read_mat_frame`, `EnvelopeMatDataset`).
- **Config:** Set `data_root` (list of paths) and optionally `mat_pattern` (default `envelope_flow*.mat`) in your YAML or via `--data_root`, `--mat_pattern`. Default data root in `train.py` is `datasets/invivo` if not set.

---

## Adding a new model

1. Clone or copy the model repo into `models/<name>/`.
2. (Recommended) Create the conda environment: `.\setup_model.ps1 <name>` then `conda activate <name>`.
3. Run the scaffolding script:
   ```bash
   python create_adapter.py --name <name>
   ```
   This generates `models/<name>/adapter.py` and `config/<name>.yaml` with TODOs.
4. Open `adapter.py`, fix the import and constructor call if needed.
5. Review the config and adjust hyperparameters (e.g. `data_root`, `image_size`, `loss`).
6. **Training:** From the project root, with the model’s conda env activated:
   ```bash
   python train.py --config config/<name>.yaml
   ```
   You can override any option from the command line, e.g. `--lr 1e-4 --batch_size 8 --num_steps 50000`.

No changes to `train.py` are required. Each model must provide an adapter with:

- **`build(args)`** → returns an `nn.Module` (wrapped in `DataParallel` by the script).
- **`forward(model, img1, img2, args)`** → returns a list of flow predictions `[(B, 2, H, W), ...]`.

---

## Adding a new loss

1. Add a function in `framework/loss.py` with signature:
   ```python
   def my_loss(image1, image2, flow_preds, **kwargs):
       # flow_preds[-1] is the final flow (B, 2, H, W)
       return (loss_tensor, metrics_dict)
   ```
2. Register it: `LOSS_REGISTRY['my_loss'] = my_loss`
3. Set `loss: my_loss` in your config or use `--loss my_loss`.

**Built-in losses:** `ncc` (normalized cross-correlation), `photometric` (Charbonnier). All support `smooth_weight` and `mag_weight` via config.

---

## Config and CLI

- The **config file** (e.g. `config/<model>.yaml`) sets default values. **CLI flags override** any field when provided. The default config path in `train.py` is `config/config.yaml` if you don’t pass `--config`.
- **Important options:**
  - `name`, `model`, `loss`, `restore_ckpt`, `gpus`, `mixed_precision`
  - `lr`, `num_steps`, `batch_size`, `image_size`, `clip`, `wdecay`, `epsilon`, `add_noise`
  - `smooth_weight`, `mag_weight`
  - `data_root`, `mat_pattern`, `val_ratio`, `log_compress`, `cache_max`, `num_workers`, `no_persistent_workers`
  - `checkpoint_freq`, `vis_freq`
  - Model-specific (e.g. `small`, `iters`, `dropout`, `alternate_corr` for RAFT; `vxm_features`, `vxm_int_steps` for VoxelMorph)

---

## Runs and outputs

- **Run directory:** `runs/<model>_<name>_<timestamp>/`
  - `config.yaml` — snapshot of the full config used
  - `checkpoints/` — `{step}_{name}.pth` and `{name}_final.pth`
  - `vis/` — flow visualizations when `vis_freq > 0`
- **TensorBoard:** logs go to the same run directory. Use `tensorboard --logdir runs/` to compare runs.

---

## Project structure

```
train.py                  # Entry point (model-agnostic)
config/
  <model>.yaml            # Per-model config (created by create_adapter.py)
framework/
  datasets.py             # .mat envelope loader (EnvelopeMatDataset, build_dataloaders)
  loss.py                 # Loss functions + LOSS_REGISTRY
  visualization.py        # Flow visualization (save_flow_figure)
models/
  <name>/
    adapter.py            # build(args) + forward(model, img1, img2, args)
    ...                   # Original model code (unchanged)
create_adapter.py         # Scaffold adapter + config for a new model
setup_model.ps1           # Create conda env per model, PyTorch (CUDA), deps (Windows)
```

---

## Notes

- **NCC and noise:** With `loss: ncc`, input noise augmentation is disabled (`add_noise` has no effect) to preserve speckle correlation (`train.py` sets `use_noise` accordingly).
- **Resume:** Use `--restore_ckpt path/to/checkpoint.pth` to load weights before training.
- **Multi-GPU:** Set `gpus: [0, 1, ...]` in config; the trainer wraps the model in `DataParallel`.
- **No models included:** You must add at least one model under `models/<name>/` and run `setup_model.ps1` (and optionally `create_adapter.py`) before training.
