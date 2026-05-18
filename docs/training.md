# Training

## Basic usage

```bash
python scripts/train_classifier.py \
    --config config/train_ast.yaml \
    --data-root $SCRATCH/Audio-XAI/audio_xai/data/external \
    --train-csv $SCRATCH/Audio-XAI/audio_xai/data/external/train.csv \
    --val-csv   $SCRATCH/Audio-XAI/audio_xai/data/external/valid.csv
```

CLI flags override YAML values, so you can keep a base config and tweak individual parameters without editing the file.

## Config files

| Config | Model | Notes |
|--------|-------|-------|
| `config/train_ast.yaml` | AST | 5s clips, batch 64 |
| `config/train_vggish.yaml` | VGGish | Requires `--vggish-ckpt` |

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `ast` | `ast` or `vggish` |
| `--clip-seconds` | `10.0` | Clip length (affects AST position embeddings) |
| `--batch-size` | `8` | Per-GPU batch size |
| `--epochs` | `10` | Training epochs |
| `--lr` | `1e-4` | Adam learning rate |
| `--train-csv` / `--val-csv` | — | Pre-split CSVs (preferred) |
| `--real-subdir` / `--fake-subdir` | `real_songs` / `fake_songs` | Fallback directory mode |
| `--val-frac` | `0.1` | Validation fraction (directory mode only) |
| `--max-per-class` | `None` | Cap samples per class |
| `--vggish-ckpt` | — | Required for VGGish (`vggish_model.ckpt` path) |
| `--num-nodes` | `1` | SLURM nodes |
| `--devices` | `1` | GPUs per node |
| `--strategy` | `auto` | Lightning strategy (`ddp_find_unused_parameters_false` for multi-GPU) |
| `--seed` | `42` | Reproducibility seed |

## Multi-GPU training

```bash
python scripts/train_classifier.py \
    --config config/train_ast.yaml \
    --data-root ... \
    --num-nodes 2 --devices 2 \
    --strategy ddp_find_unused_parameters_false
```

On SLURM, submit via `sbatch sbatch/train/train_classifier_new_ast.sbatch` instead — the sbatch script sets `--nodes`, `--ntasks-per-node`, and SLURM environment variables automatically.

## Checkpoints and logs

Checkpoints are saved under `--out-dir` (default `runs/`):

```
runs/
└── ast/
    └── version_N/
        ├── checkpoints/
        │   ├── epoch=4-step=9740.ckpt    # best val/eer
        │   └── 010.ckpt                  # periodic save every 10 epochs
        └── events.out.tfevents.*          # TensorBoard logs
```

TensorBoard: `tensorboard --logdir runs/`

The best checkpoint is selected by lowest **EER** (Equal Error Rate) on the validation split. Two best checkpoints are kept.

## Class imbalance

When using directory-mode splits, `stratified_split` computes inverse-frequency class weights passed to the loss function. When using CSV splits (pre-balanced 50/50), uniform weights are used.

## AST clip length

`--clip-seconds` controls both the mel frame count and the AST backbone's `max_length` (position embedding size). Training and inference **must use the same `clip_seconds`** — mismatches cause a shape error. The value is stored in the Lightning checkpoint and passed automatically when loading.

## Monitoring training

```bash
tensorboard --logdir $SCRATCH/Audio-XAI/runs/ast
```

Logged metrics per epoch: `train/loss`, `val/loss`, `val/acc`, `val/auroc`, `val/f1`, `val/eer`, `val/sensitivity`, `val/specificity`.