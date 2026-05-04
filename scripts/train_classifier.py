"""Training entry point with SLURM multi-node support.

Single-node (interactive):
    python scripts/train_classifier.py \\
        --data-root /path/to/sonics \\
        --model ast \\
        --batch-size 8 \\
        --epochs 10

Multi-node SLURM (via sbatch):
    srun python scripts/train_classifier.py \\
        --data-root /path/to/sonics \\
        --model ast \\
        --num-nodes 2 \\
        --devices 4

    SBATCH directives must match:
        #SBATCH --nodes=2              # == --num-nodes
        #SBATCH --ntasks-per-node=4    # == --devices
        #SBATCH --gres=gpu:4
"""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader, random_split

from audio_xai.data.sonics import SonicsConfig, SonicsDataset
from audio_xai.models.ast_binary import ASTBinary
from audio_xai.models.lit_module import RealFakeLitModule
from audio_xai.models.vggish_binary import VGGishBinary


def build_model(name: str):
    if name == "ast":
        return ASTBinary(pretrained=True)
    if name == "vggish":
        return VGGishBinary()
    raise ValueError(f"Unknown model: {name!r}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--real-subdir", type=str, default="real")
    p.add_argument("--fake-subdir", type=str, default="fake")
    p.add_argument("--model", choices=["ast", "vggish"], default="ast")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip-seconds", type=float, default=10.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    # ── Cluster args ──────────────────────────────────────────────────────────
    p.add_argument("--num-nodes", type=int, default=1,
                   help="Number of SLURM nodes — must match #SBATCH --nodes")
    p.add_argument("--devices", default="auto",
                   help="GPUs per node ('auto' or int) — must match #SBATCH --ntasks-per-node")
    p.add_argument("--strategy", type=str, default="auto",
                   help="Lightning strategy: 'auto', 'ddp', 'ddp_find_unused_parameters_false'")
    p.add_argument("--no-auto-requeue", action="store_true",
                   help="Disable SLURM auto-requeue on wall-time limit")
    args = p.parse_args()

    seed_everything(args.seed)

    cfg = SonicsConfig(
        root=args.data_root,
        clip_seconds=args.clip_seconds,
        real_subdir=args.real_subdir,
        fake_subdir=args.fake_subdir,
    )
    full = SonicsDataset(cfg)
    n_val = int(len(full) * args.val_frac)
    n_train = len(full) - n_val
    train_set, val_set = random_split(full, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.model)
    lit = RealFakeLitModule(model, lr=args.lr)

    logger = TensorBoardLogger(save_dir=str(args.out_dir), name=args.model)
    ckpt = ModelCheckpoint(monitor="val/eer", mode="min", save_top_k=2)

    # SLURMEnvironment handles node discovery and auto-requeue on wall-time
    # signal (SIGUSR1 sent 90 s before the job is killed by the scheduler).
    slurm_plugin = SLURMEnvironment(
        requeue_signal=signal.SIGUSR1,
        auto_requeue=not args.no_auto_requeue,
    )

    devices = args.devices if args.devices == "auto" else int(args.devices)

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[ckpt],
        accelerator="gpu",
        devices=devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        precision="16-mixed",
        plugins=[slurm_plugin],
    )
    trainer.fit(lit, train_loader, val_loader)


if __name__ == "__main__":
    main()