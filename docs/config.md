# Config reference

All scripts accept a `--config` YAML file. CLI flags always override YAML values.

## How configs are loaded

```python
# Pre-parsed before argparse so YAML values become defaults:
python scripts/train_classifier.py --config config/train_ast.yaml --lr 5e-5
# → lr=5e-5 (CLI wins), everything else from YAML
```

## `config/train_ast.yaml` / `train_vggish.yaml`

```yaml
seed: 42
clip_seconds: 5.0          # audio clip length in seconds

model:
  name: ast                # ast | vggish
  vggish_ckpt: null        # path to vggish_model.ckpt (required for vggish)

data:
  root: null               # set via --data-root
  train_csv: null          # CSV with filepath/target columns
  val_csv: null
  real_subdir: real_songs  # directory-mode fallback
  fake_subdir: fake_songs
  val_frac: 0.1
  max_per_class: null

training:
  batch_size: 64
  epochs: 10
  lr: 1.0e-4
  num_workers: 16
  out_dir: runs

cluster:
  num_nodes: 1
  devices: 1
  strategy: auto           # ddp_find_unused_parameters_false for multi-GPU
  no_auto_requeue: false
```

## `config/predict_ast.yaml` / `predict_vggish.yaml` / `predict_sonics.yaml`

```yaml
seed: 42
clip_seconds: 5.0

model:
  name: ast
  checkpoint: null         # set via --checkpoint or AST_CKPT env var
  model_id: null           # HuggingFace repo ID (Sonics only)

predict:
  split: test              # train | valid | test
  n_samples: null          # null = full split; integer = balanced cap
  batch_size: 16
  sample_rate: 16000
  log_dir: runs/predict
  device: cuda

attack:
  n_attack_samples: 100
  micro_batch: 4           # --attack-micro-batch
  n_steps: 100             # Adam optimisation steps
  lr: 1.0e-3
  lambda_aud: 0.5          # audibility loss weight
  lambda_pred: 100.0       # prediction preservation weight
  pred_margin: 0.1         # hinge margin
  log_dir: runs/attack
  device: cuda
```

## AttackConfig fields

These map to the `AttackConfig` dataclass used internally by the attack engine:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_steps` | int | 200 | Adam steps per sample |
| `lr` | float | 1e-3 | Adam learning rate |
| `lambda_audibility` | float | 1.0 | Weight on the inaudibility loss term |
| `lambda_pred` | float | 100.0 | Weight on the prediction-preservation loss |
| `pred_margin` | float | 1.0 | Hinge margin — lower activates penalty earlier |
| `linf_bound` | float | 0.01 | Hard L∞ clip applied to δ after each step |
| `sample_rate` | int | 16000 | Audio sample rate |
| `use_gradient_checkpointing` | bool | True | Recompute activations during backprop (saves VRAM) |
| `log_every` | int\|None | 20 | Log loss every N steps; None disables |

## Tuning guidance

| Goal | Recommendation |
|------|---------------|
| Stronger explanation flip | Decrease `lambda_pred`, increase `n_steps` |
| Keep predictions intact | Increase `lambda_pred`, increase `pred_margin` |
| More inaudible perturbation | Increase `lambda_aud` |
| Faster runs (less OOM risk) | Decrease `attack-micro-batch`, decrease `n_steps` |
| Reduce memory on AST | `clip_seconds: 2.0` — drops tokens from 1214 → 230 |