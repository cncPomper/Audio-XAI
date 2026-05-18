# SLURM / PLGrid guide

The project runs on **PLGrid Athena** (partition `plgrid-gpu-a100`, account `plggolemml26-gpu-a100`). All sbatch scripts live under `$SCRATCH/sbatch/`.

## Directory structure

```
sbatch/
├── train/
│   ├── train_classifier_new_ast.sbatch
│   └── train_classifier_new_vggish.sbatch
├── predict/
│   ├── ast.sbatch
│   ├── vggish.sbatch
│   └── spectttra.sbatch
└── attack/
    ├── ast.sbatch
    └── vggish.sbatch
```

## Submitting jobs

```bash
cd $SCRATCH

# Training
sbatch sbatch/train/train_classifier_new_ast.sbatch

# Inference
sbatch sbatch/predict/ast.sbatch

# Attack
sbatch sbatch/attack/ast.sbatch
```

## Monitoring

```bash
# List your jobs
squeue -u $USER

# Watch a running job's output
tail -f $SCRATCH/Audio-XAI/logs/<job_name>-<job_id>.out

# Cancel a job
scancel <job_id>
```

Log files are written to `$SCRATCH/Audio-XAI/logs/`:

```
logs/
├── ast_attack-2581846.out
└── ast_attack-2581846.err
```

## Resource allocation per script

| Script | Nodes | GPUs | RAM | Time |
|--------|-------|------|-----|------|
| `train/ast.sbatch` | 2 | 2×2 A100 | 128 GB | 12 h |
| `train/vggish.sbatch` | 2 | 2×2 A100 | 128 GB | 12 h |
| `predict/ast.sbatch` | 1 | 1 A100 | 64 GB | 6 h |
| `predict/vggish.sbatch` | 1 | 1 A100 | 32 GB | 6 h |
| `attack/ast.sbatch` | 1 | 1 A100 | 64 GB | 12 h |
| `attack/vggish.sbatch` | 1 | 1 A100 | 32 GB | 6 h |

## Environment variables

Override defaults without editing sbatch files:

```bash
# Use a different checkpoint
AST_CKPT=$SCRATCH/Audio-XAI/runs/ast/version_7/checkpoints/best.ckpt \
    sbatch sbatch/attack/ast.sbatch

# Run on CPU (debugging)
DEVICE=cpu sbatch sbatch/predict/ast.sbatch
```

## Batched attack jobs

The attack sbatch scripts split `n_attack_samples` across `N_BATCHES` sequential chunks within a single job:

```bash
# In sbatch/attack/ast.sbatch:
N_BATCHES=8
for BATCH_IDX in $(seq 0 $((N_BATCHES - 1))); do
    srun python3 scripts/attack.py \
        --n-batches $N_BATCHES --batch-index $BATCH_IDX \
        --run-name  "$RUN_NAME" ...
done
```

All batches write to the same `RUN_NAME` directory. If a batch fails, re-submitting the job skips already-processed samples automatically (resume logic checks for `audio/*/adversarial.wav`).

## Multi-node training

Training uses Lightning's `SLURMEnvironment` plugin which handles:
- Node discovery via `SLURM_NODELIST`
- Auto-requeue on wall-time limit (`--no-auto-requeue` disables this)
- DDP process group setup

Set `--strategy ddp_find_unused_parameters_false` for AST (some transformer parameters are unused in forward).

## TensorBoard on the cluster

TensorBoard cannot be accessed directly from a compute node. Use SSH port forwarding:

```bash
# On your local machine:
ssh -L 6006:localhost:6006 plgpkitlo@athena.cyfronet.pl \
    "tensorboard --logdir $SCRATCH/Audio-XAI/runs --port 6006"
```

Then open `http://localhost:6006` in your browser.

## Common issues

**Job killed immediately:** Check the `.err` log — usually a missing module or wrong conda env.

**OOM on attack:** Reduce `--attack-micro-batch` or `--clip-seconds`. The sbatch scripts set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` which helps with fragmentation.

**NCCL errors on multi-node:** Ensure `--strategy ddp_find_unused_parameters_false` is set and the number of `--ntasks-per-node` matches `--devices`.

**HuggingFace download fails:** Check that `$SCRATCH/hf_token` exists and contains a valid token. Verify `HF_HOME` points to scratch (not home directory).