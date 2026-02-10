## Overview

This repository provides a framework for training and running displacement / flow models (for example RAFT) using YAML config files.

## Requirements

- **conda** (recommended) with Python 3.10 or later
- **GPU** is optional (CUDA or Apple MPS if available); CPU will work but be slower

## Quick start (RAFT example)

1. Open a terminal in the root of this repo.
2. Make sure the sample `.mat` files are present in the `data/` directory (already included for the RAFT config).
3. Create the conda environment and install dependencies:

```bash
bash scripts/setup.sh configs/raft.yaml
```

4. Activate the environment:

```bash
conda activate raft
```

5. Start training:

```bash
python train.py --config configs/raft.yaml
```

## Run inference

With the same environment active:

```bash
python inference.py --config configs/raft.yaml
```

By default, displacement fields and optional visualizations are written under `outputs/raft_disp` (see `output.inference` in `configs/raft.yaml`).

## Visualize displacement fields

To visualize `.mat` flow files produced by inference or training:

```bash
python scripts/visualize.py --mat outputs/raft_disp/envelope_flow41__envelope_flow42_flow.mat
```

You can also visualize a combined `raft_disp_all.mat` file and choose a pair index:

```bash
python scripts/visualize.py --mat outputs/raft_disp/raft_disp_all.mat --pair_idx 0 --stride 20
```

## Using a different model

- **Create a config**: add a YAML file under `configs/` that sets `model:` and a `setup:` section with its dependencies.
- **Set up the env**: run `bash scripts/setup.sh <path_to_new_config>`.
- **Train / infer**: run `train.py` and `inference.py` with `--config <path_to_new_config>`.

## How to write a config file

The easiest way is to copy an existing config (for example `configs/raft.yaml` or `configs/voxelmorph.yaml`) and change only what you need.

At minimum, a config should define:

- **model**: short model name, for example:

```yaml
model: mymodel
```

- **setup**: Python version and dependencies to install into the conda env:

```yaml
setup:
  python: "3.10"
  conda:
    - "numpy"
    - "scipy"
  pip:
    - "torch"
    - "torchvision"
```

- **input**: where the data lives and how to read it:

```yaml
input:
  data_input:
    path: data
    image_type: envelope_flow*.mat
    image_key: img
```

- **models.<model_name>**: model-specific settings, read by your adapter:

```yaml
models:
  mymodel:
    model_settings:
      # your model options here
```

- **training.<model_name>**: training hyperparameters:

```yaml
training:
  mymodel:
    batch_size: 4
    lr: 1.0e-4
    num_steps: 200
```

See `configs/raft.yaml` and `configs/voxelmorph.yaml` for full examples.

## How adapters work

- **Lookup**: the framework calls `load_adapter(model_name)` from `framework/registry.py`.
- **Search order**:
  - first tries `framework/adapters/<model_name>.py` with a class named `Adapter`
  - if that is missing, it tries `models/<model_name>/adapter.py` with `Adapter`, `<MODEL>Adapter`, or `<Model>Adapter`
- **Required methods**:
  - `build(cfg, repo_root) -> torch.nn.Module`: build and return the model
  - `forward(model, batch, cfg) -> dict`: run the model and return at least `disp_x`, `disp_y`, and any visualization tensors
  - optionally `compute_loss(...)` for training, if you want to use `train.py`

To add a new model:

1. Put the third-party code under `models/<model_name>/` (or use an existing installed package).
2. Create `framework/adapters/<model_name>.py` (or `models/<model_name>/adapter.py`) with an `Adapter` class that follows the RAFT and VoxelMorph adapters as templates.
3. Create `configs/<model_name>.yaml` as devscribed above, then run `scripts/setup.sh`, `train.py`, and `inference.py` with that config.

## Template files you can copy

- **Base config template**: `configs/base_model.yaml`
  - Copy to `configs/<your_model_name>.yaml`.
  - Rename `model: base_model` to `model: <your_model_name>`.
  - Update `models.base_model` and `training.base_model` keys to match your model name.
- **Base adapter template**: `framework/adapters/base_model.py`
  - Copy to `framework/adapters/<your_model_name>.py`.
  - Implement `build`, `forward`, and `compute_loss` following the comments in the file.

After copying and editing these two templates, you can run:

```bash
bash scripts/setup.sh configs/<your_model_name>.yaml
conda activate <your_model_name>
python train.py --config configs/<your_model_name>.yaml
python inference.py --config configs/<your_model_name>.yaml
```
