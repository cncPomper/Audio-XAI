# Audio-XAI

Audio-XAI studies the perceptual fragility of explanation methods (Grad-CAM, LRP) for deep-learning audio classifiers. The core question: can an adversary flip what the model *explains* while keeping predictions identical, and can that perturbation remain inaudible?

**Models supported:** AST, VGGish, Sonics (SpecTTTra)
**XAI methods:** Grad-CAM (CNN + Transformer variants), LRP
**Perceptual metrics:** PESQ, STOI, PEAQ, ViSQOL, CDPAM
**Infrastructure:** PyTorch Lightning, HuggingFace, SLURM/PLGrid A100 cluster

## Docs

| Page | Contents |
|------|----------|
| [Setup](setup.md) | Install, cluster environment, conda env |
| [Data](data.md) | Dataset layout, CSV format, SonicsDataset |
| [Training](training.md) | Train classifiers locally or on SLURM |
| [Inference](inference.md) | Batch prediction, metrics, checkpoints |
| [Attack](attack.md) | Perceptual XAI attack — full workflow |
| [Config reference](config.md) | Every YAML key explained |
| [SLURM guide](slurm.md) | Submitting and monitoring jobs on PLGrid |
| [API Reference](api.md) | Auto-generated module docs |

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/cncPomper/Audio-XAI && cd Audio-XAI
conda activate $SCRATCH/conda_envs/athena   # or: uv sync

# 2. Train
python scripts/train_classifier.py --config config/train_ast.yaml --data-root /path/to/data

# 3. Predict
python scripts/predict.py --config config/predict_ast.yaml \
    --data-root /path/to/data --checkpoint runs/ast/version_0/checkpoints/epoch=4.ckpt

# 4. Attack
python scripts/attack.py --config config/predict_ast.yaml \
    --data-root /path/to/data --checkpoint runs/ast/version_0/checkpoints/epoch=4.ckpt \
    --full-audio --window-hop-seconds 5.0
```

## Repository layout

```
Audio-XAI/
├── config/             # YAML configs for train / predict / attack
├── scripts/            # Entry-point scripts (train, predict, attack, explain)
├── audio_xai/          # Main Python package
│   ├── models/         # ASTBinary, VGGishBinary, Wav2Vec2Binary + LightningModule
│   ├── attacks/        # perceptual_xai_attack, AttackConfig, AttackResult
│   ├── data/           # SonicsDataset, SonicsConfig
│   ├── xai/            # GradCAM (CNN + Transformer), LRP
│   └── metrics/        # psychoacoustic masking, PESQ/STOI wrappers
├── docs/               # This documentation
└── sbatch/             # SLURM job scripts (train / predict / attack)
    ├── train/
    ├── predict/
    └── attack/
```
