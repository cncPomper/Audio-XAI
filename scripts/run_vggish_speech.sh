#!/usr/bin/env bash
# Experiment: VGGish on Speech Commands
# Set VGGISH_CKPT to the path of vggish_model.ckpt before running.
# Download: https://storage.googleapis.com/audioset/vggish_model.ckpt

#SBATCH --job-name=xai_vggish_speech
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=logs/vggish_speech_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=piotr.kitlowski@gmail.com

ml Miniconda3/25.7.0-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCRATCH/conda_envs/athena"

SPEECH_DIR="${SPEECH_DIR:-speech_commands_samples}"
OUT_DIR="${OUT_DIR:-reports/speech_vggish}"
N_SAMPLES="${N_SAMPLES:-100}"
ATTACK_STEPS="${ATTACK_STEPS:-200}"
VGGISH_CKPT="${VGGISH_CKPT:-}"   # set this before running

VGGISH_ARG=""
if [ -n "$VGGISH_CKPT" ]; then
    VGGISH_ARG="--vggish-ckpt $VGGISH_CKPT"
fi

python scripts/run_speech_experiment.py \
    --speech-dir "$SPEECH_DIR" \
    --model vggish \
    $VGGISH_ARG \
    --n-samples "$N_SAMPLES" \
    --attack-steps "$ATTACK_STEPS" \
    --clip-seconds 1.0 \
    --linf-bound 0.01 \
    --lambda-aud 1.0 \
    --lambda-pred 100.0 \
    --out-dir "$OUT_DIR"