#!/usr/bin/env bash
# Fine-tune AST / VGGish / SpecTra on Google Speech Commands.
# Usage:
#   sbatch scripts/train.sh ast
#   sbatch scripts/train.sh vggish /path/to/vggish_model.ckpt
#   sbatch scripts/train.sh spectra

#SBATCH --job-name=train_speech
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/train_speech_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=piotr.kitlowski@gmail.com

ml Miniconda3/25.7.0-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCRATCH/conda_envs/athena"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${1:?Usage: sbatch train.sh <ast|vggish|spectra> [vggish_ckpt]}"
VGGISH_CKPT="${2:-}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
VERSION="${VERSION:-v0.02}"
OUT_DIR="${OUT_DIR:-runs/speech_${MODEL}}"

EXTRA_ARGS=""
if [ -n "$VGGISH_CKPT" ]; then
    EXTRA_ARGS="--vggish-ckpt $VGGISH_CKPT"
fi

python "$SCRIPT_DIR/train_speech_classifier.py" \
    --model      "$MODEL"      \
    --version    "$VERSION"    \
    --epochs     "$EPOCHS"     \
    --batch-size "$BATCH_SIZE" \
    --lr         "$LR"         \
    --out-dir    "$OUT_DIR"    \
    --devices    2             \
    --strategy   "ddp"         \
    $EXTRA_ARGS