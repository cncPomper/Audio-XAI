# Inference

## Basic usage

```bash
python scripts/predict.py \
    --config config/predict_ast.yaml \
    --data-root $SCRATCH/Audio-XAI/audio_xai/data/external \
    --checkpoint $SCRATCH/Audio-XAI/runs/ast/version_5/checkpoints/epoch=4-step=9740.ckpt
```

## Models

### AST

```bash
python scripts/predict.py \
    --model-type ast \
    --checkpoint runs/ast/version_5/checkpoints/epoch=4-step=9740.ckpt \
    --data-root /path/to/data --split test
```

### VGGish

```bash
python scripts/predict.py \
    --model-type vggish \
    --checkpoint runs/vggish/version_3/checkpoints/epoch-epoch=009.ckpt \
    --data-root /path/to/data --split test
```

### Sonics (HuggingFace)

```bash
python scripts/predict.py \
    --model-type sonics \
    --model-id awsaf49/sonics-spectttra-gamma-5s \
    --data-root /path/to/data --split test
```

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-type` | — | `ast`, `vggish`, or `sonics` |
| `--model-id` | — | HuggingFace repo (Sonics only) |
| `--checkpoint` | — | Path to `.ckpt` file (AST / VGGish) |
| `--data-root` | — | Dataset root directory |
| `--split` | `test` | `train`, `valid`, or `test` |
| `--n-samples` | `None` | Cap to N balanced samples (None = full split) |
| `--clip-seconds` | `5.0` | Clip duration |
| `--batch-size` | `16` | Inference batch size |
| `--log-dir` | `runs/predict` | TensorBoard output directory |
| `--device` | `cuda` | `cuda` or `cpu` |

## Output

Results are saved to `--log-dir`:

- **TensorBoard scalars:** accuracy, AUROC, F1, EER (per-sample and aggregate)
- **CSV:** `results.csv` with per-sample predictions, probabilities, and true labels

## Metrics computed

| Metric | Description |
|--------|-------------|
| Accuracy | Fraction of correct predictions |
| AUROC | Area under ROC curve |
| F1 (macro) | Macro-averaged F1 score |
| EER | Equal Error Rate — the standard deepfake metric |
| Sensitivity | True positive rate (fake detected as fake) |
| Specificity | True negative rate (real detected as real) |

## SLURM

```bash
sbatch sbatch/predict/ast.sbatch
sbatch sbatch/predict/vggish.sbatch
sbatch sbatch/predict/spectttra.sbatch
```

Override the checkpoint or split with environment variables before submitting:

```bash
AST_CKPT=/path/to/other.ckpt sbatch sbatch/predict/ast.sbatch
DEVICE=cpu sbatch sbatch/predict/ast.sbatch
```