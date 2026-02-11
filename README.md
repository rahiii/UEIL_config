## Overview

Framework for training and running displacement / flow models using YAML configs.

## Requirements

- **conda** with Python 3.10+
- GPU optional (CUDA or Apple MPS); CPU works but slower

## Add a new model

```bash
python scripts/add_model.py
```

Answer 5 questions. It generates `configs/<name>.yaml`. No Python adapter file needed. Then:

```bash
bash scripts/setup.sh configs/<name>.yaml
conda activate <name>
python train.py --config configs/<name>.yaml
python inference.py --config configs/<name>.yaml
```

## How the config works

The config drives everything. You write the model name **once**. Every field maps to a specific file:

| Config key | What it does | Read by |
|---|---|---|
| `model` | Name for logging / lookups | `train.py`, `inference.py` |
| `adapter.entry_point` | Python class to import | `framework/adapter.py` → `_import_class()` |
| `adapter.build_args` | Kwargs passed to constructor | `framework/adapter.py` → `GenericAdapter.build()` |
| `adapter.forward.input` | What tensors to pass to model | `framework/adapter.py` → `GenericAdapter.forward()` |
| `adapter.forward.output` | How to extract flow from result | `framework/adapter.py` → `_extract_flow()` |
| `adapter.forward.output_format` | Shape of flow tensor | `framework/adapter.py` → `_normalize_flow()` |
| `setup` | Conda/pip deps | `scripts/setup.sh` |
| `input.data_input` | Data path, file pattern, resize | `framework/datasets.py` |
| `input.conversion.view` | Tensor layout sent to model | `framework/datasets.py` → `apply_conversion()` |
| `training.lr`, `.batch_size`, etc. | Training hyperparams | `train.py` |
| `training.loss` | Loss function + weights | `framework/loss.py` → `compute_loss()` |
| `output.inference` | Where/how to save results | `inference.py` |

## Adapter input modes

| Mode | Batch keys | Model call |
|---|---|---|
| `pair_rgb` | `img1 [B,3,H,W]`, `img2 [B,3,H,W]` | `model(img1, img2)` |
| `pair_gray` | `img1 [B,1,H,W]`, `img2 [B,1,H,W]` | `model(img1, img2)` |
| `stack2ch` | `x [B,2,H,W]` | `model(x)` |
| `stack6ch` | `x [B,6,H,W]` | `model(x)` |
| `video_tchw` | `x [B,T,C,H,W]` | `model(x)` |

## Adapter output modes

| Mode | Model returns | Framework takes |
|---|---|---|
| `direct` | flow tensor | as-is |
| `list_last` | `[iter1, ..., final]` | `[-1]` |
| `list_first` | `[flow, ...]` | `[0]` |
| `tuple_first` | `(flow, extras)` | `[0]` |
| `tuple_index:N` | `(a, b, c)` | `[N]` |
| `dict_key:K` | `{"K": flow}` | `["K"]` |

## Custom adapter (escape hatch)

If the generic adapter can't handle your model, create `models/<name>/adapter.py`:

```python
class Adapter:
    def build(self, cfg, repo_root):
        from models.mymodel.net import MyNet
        return MyNet()

    def forward(self, model, batch, cfg):
        device = next(model.parameters()).device
        img1 = batch["img1"].to(device)
        img2 = batch["img2"].to(device)
        flow = model(img1, img2)
        return {
            "disp_x": flow[:, 0].cpu(),
            "disp_y": flow[:, 1].cpu(),
            "flow_device": flow,
            "flow_vis_rgb": ...,  # [B,3,H,W] uint8
        }
```

Remove the `adapter:` block from your config — the file-based adapter is only used when the config has no `adapter.entry_point`.

## Visualize

```bash
python scripts/visualize.py --mat outputs/<model>_disp/<file>_flow.mat
python scripts/visualize.py --mat outputs/<model>_disp/<model>_disp_all.mat --pair_idx 0
```

## Project structure

```
configs/             YAML configs (one per model)
framework/
  adapter.py         Config-driven adapter + loader
  datasets.py        Data loading, normalization, tensor conversion
  loss.py            Loss functions (photometric, NCC, MSE, etc.)
models/              Model code (cloned repos or your own)
scripts/
  add_model.py       Interactive scaffolding
  setup.sh           Conda env + dependency installer
  visualize.py       Displacement field plotting
train.py             Training loop
inference.py         Inference loop
```
