#!/usr/bin/env bash
# Experiment: AST on Speech Commands
# Adjust SPEECH_DIR and OUT_DIR before running.
# For SLURM: sbatch scripts/run_ast_speech.sh

#SBATCH --job-name=xai_ast_speech
#SBATCH --account=plggolemml26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/ast_speech_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=piotr.kitlowski@gmail.com

ml Miniconda3/25.7.0-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCRATCH/conda_envs/athena"

SPEECH_DIR="${SPEECH_DIR:-speech_commands_samples}"
OUT_DIR="${OUT_DIR:-reports/speech_ast}"
N_SAMPLES="${N_SAMPLES:-100}"
ATTACK_STEPS="${ATTACK_STEPS:-200}"

python scripts/run_speech_experiment.py \
    --speech-dir "$SPEECH_DIR" \
    --model ast \
    --n-samples "$N_SAMPLES" \
    --attack-steps "$ATTACK_STEPS" \
    --clip-seconds 1.0 \
    --linf-bound 0.01 \
    --lambda-aud 1.0 \
    --lambda-pred 100.0 \
    --out-dir "$OUT_DIR"
