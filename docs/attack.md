# Adversarial XAI Attack

The perceptual XAI attack optimises a perturbation `δ` added to an audio signal so that:

1. **Explanation flips** — the Grad-CAM heatmap on `x + δ` has low cosine similarity with the heatmap on `x`.
2. **Prediction is preserved** — the model's class decision does not change.
3. **Perturbation is inaudible** — `δ` stays below the psychoacoustic masking threshold.

## Basic usage

```bash
python scripts/attack.py \
    --config config/predict_ast.yaml \
    --data-root $SCRATCH/Audio-XAI/audio_xai/data/external \
    --checkpoint $SCRATCH/Audio-XAI/runs/ast/version_5/checkpoints/epoch=4-step=9740.ckpt \
    --full-audio \
    --window-hop-seconds 5.0
```

## Loss function

```
L = L_explanation  +  λ_aud · L_audibility  +  λ_pred · L_prediction
```

| Term | Description |
|------|-------------|
| `L_explanation` | `1 − cosine_similarity(CAM(x), CAM(x+δ))` — maximise disagreement |
| `L_audibility` | Perturbation power above psychoacoustic masking threshold |
| `L_prediction` | Hinge loss penalising prediction flips (`pred_margin` gap) |

`δ` is additionally hard-clipped to `linf_bound = 0.01` after each step.

## Full-audio vs clip mode

| Mode | Flag | Description |
|------|------|-------------|
| Clip | (default) | Attack a single fixed-length clip |
| Full audio | `--full-audio` | Slide a window across the entire file, stitch deltas |

Full-audio mode uses `--window-hop-seconds` to control overlap. A hop of 5 s on 10 s clips gives 50% overlap. Use `--attack-micro-batch` to control how many windows are processed simultaneously on GPU.

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--n-attack-samples` | from config | Number of samples to attack (balanced real/fake) |
| `--n-steps` | `50` | Adam optimisation steps per sample |
| `--lr` | `1e-3` | Adam learning rate |
| `--lambda-aud` | `1.0` | Audibility loss weight |
| `--lambda-pred` | `100.0` | Prediction preservation weight |
| `--pred-margin` | `1.0` | Hinge margin (lower activates penalty earlier) |
| `--full-audio` | off | Attack the full waveform via sliding windows |
| `--window-hop-seconds` | `clip_seconds` | Window hop for full-audio mode |
| `--attack-micro-batch` | `1` | Windows processed per GPU step (increase if VRAM allows) |
| `--n-batches` | `1` | Split samples across N sequential runs |
| `--batch-index` | `0` | Index of this run (0-based) |
| `--run-name` | auto | Shared output directory name across batches |
| `--oom-retries` | `3` | GPU OOM retries before skipping a sample |
| `--log-dir` | `runs/attack` | Output root directory |

## Output structure

```
runs/attack/{run_name}/
├── sample_0000_{stem}.json      # per-sample metrics
├── sample_0001_{stem}.json
├── audio/
│   └── {stem}/
│       ├── original.wav
│       ├── adversarial.wav
│       └── delta.wav
├── heatmaps/
│   └── {stem}/
│       ├── original.npy         # Grad-CAM on original
│       └── adversarial.npy      # Grad-CAM on adversarial
└── images/
    └── {stem}/                  # PNG visualisations
```

Each `sample_XXXX.json` contains:

```json
{
  "index": 0,
  "stem": "fake_00001_suno_0",
  "label": 1,
  "pred_orig": 1,
  "pred_adv": 1,
  "prob_orig": 0.97,
  "prob_adv": 0.91,
  "cos_sim": 0.23,
  "top10_overlap": 0.14,
  "pred_preserved": true,
  "delta_linf": 0.0082,
  "pesq": 3.41,
  "stoi": 0.94,
  "visqol": 4.12,
  "ok": true
}
```

## Batched runs and resuming

To spread 100 samples across 8 sequential SLURM jobs (each handles 12-13 samples):

```bash
# In ast.sbatch — this loop is already set up:
N_BATCHES=8
for BATCH_IDX in $(seq 0 $((N_BATCHES - 1))); do
    srun python3 scripts/attack.py \
        --n-batches   $N_BATCHES \
        --batch-index $BATCH_IDX \
        --run-name    "$RUN_NAME" ...
done
```

**Resume:** On each run start, the script checks `runs/attack/{run_name}/audio/*/adversarial.wav` to find already-processed stems and skips them. Resubmitting a job after partial failure continues from where it left off.

## SLURM

```bash
sbatch sbatch/attack/ast.sbatch
sbatch sbatch/attack/vggish.sbatch
```

Environment variable overrides:

```bash
AST_CKPT=/path/to/other.ckpt sbatch sbatch/attack/ast.sbatch
DEVICE=cpu sbatch sbatch/attack/ast.sbatch
```

## Memory tuning

AST with full-audio mode and second-order GradCAM is memory-intensive.

| Setting | Effect |
|---------|--------|
| `--attack-micro-batch 1` | Safest — one window at a time (~8 GiB) |
| `--attack-micro-batch 4` | ~24 GiB — faster |
| `--attack-micro-batch 8` | ~40 GiB — fills an A100 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduces fragmentation |

If a sample triggers OOM, the script halves the micro-batch and retries up to `--oom-retries` times before skipping.

## Monitoring

```bash
tensorboard --logdir $SCRATCH/Audio-XAI/runs/attack
```

Logged per sample: spectrograms (original / adversarial / delta), audio waveforms, GPU memory, cosine similarity, prediction preservation.