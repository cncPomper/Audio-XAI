# Setup

## PLGrid / Athena cluster (recommended)

The project runs on PLGrid Athena with NVIDIA A100 GPUs. The shared conda environment is already configured.

```bash
ml Miniconda3/25.7.0-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $SCRATCH/conda_envs/athena
```

Set cache locations so large model downloads land on scratch (not home quota):

```bash
export HF_TOKEN=$(cat $SCRATCH/hf_token)   # required for gated HuggingFace models
export HF_HOME=$SCRATCH/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export XDG_CACHE_HOME=$SCRATCH/.cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
```

All SLURM sbatch scripts set these automatically — you only need them for interactive sessions.

## Local development

Python 3.12+ is required.

### uv (fastest)

```bash
git clone https://github.com/cncPomper/Audio-XAI && cd Audio-XAI
uv sync          # installs all dependencies into .venv
source .venv/bin/activate
```

### conda

```bash
conda create -n audio_xai python=3.12
conda activate audio_xai
pip install -e ".[dev]"
```

## Dependencies

Key runtime dependencies (see `pyproject.toml` for pinned versions):

| Library | Purpose |
|---------|---------|
| `torch` / `torchaudio` | Core tensor ops and audio I/O |
| `lightning` | Training loop, checkpointing, SLURM integration |
| `transformers` | HuggingFace AST and Wav2Vec2 backbones |
| `torchmetrics` | Accuracy, AUROC, F1, EER |
| `tensorboard` | Experiment tracking |
| `pesq` / `pystoi` | PESQ and STOI audio quality metrics |
| `cdpam` | Deep perceptual audio metric |
| `audio-metrics` / `zimtohrli` | ViSQOL and perceptual metrics |

## VGGish checkpoint

VGGish requires a TensorFlow checkpoint file that is **not** included in the repo:

```bash
wget https://storage.googleapis.com/audioset/vggish_model.ckpt -P $SCRATCH/
```

Pass the path via `--vggish-ckpt $SCRATCH/vggish_model.ckpt` or set it in your YAML config.

## HuggingFace token

Sonics (SpecTTTra) is a gated model. Request access at HuggingFace and store your token:

```bash
echo "hf_..." > $SCRATCH/hf_token
chmod 600 $SCRATCH/hf_token
```

## Verify installation

```bash
python - <<'EOF'
import torch, torchaudio, lightning, transformers
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("torchaudio:", torchaudio.__version__)
print("lightning:", lightning.__version__)
print("transformers:", transformers.__version__)
EOF
```
