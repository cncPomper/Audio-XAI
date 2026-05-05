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
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader

from audio_xai.data.sonics import SonicsConfig, SonicsDataset
from audio_xai.models.ast_binary import ASTBinary
from audio_xai.models.lit_module import RealFakeLitModule
from audio_xai.models.vggish_binary import VGGishBinary


def stratified_split(dataset: SonicsDataset, val_frac: float) -> tuple[Subset, Subset, torch.Tensor]:
    """Split dataset preserving class ratios; return (train, val, class_weights).

    class_weights[c] = total / (n_classes * count[c]) — inverse-frequency weighting
    for CrossEntropyLoss to handle class imbalance.
    """
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, (_, label) in enumerate(dataset._samples):
        label_to_indices[label].append(i)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in label_to_indices.values():
        perm = torch.randperm(len(indices)).tolist()
        shuffled = [indices[j] for j in perm]
        n_val = max(1, int(len(shuffled) * val_frac))
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    n_classes = len(label_to_indices)
    counts = torch.zeros(n_classes)
    for label, indices in label_to_indices.items():
        n_train = len(indices) - max(1, int(len(indices) * val_frac))
        counts[label] = n_train
    class_weights = counts.sum() / (n_classes * counts.clamp(min=1))

    return Subset(dataset, train_indices), Subset(dataset, val_indices), class_weights


def build_model(name: str, vggish_ckpt: str | None = None):
    if name == "ast":
        return ASTBinary(pretrained=True)
    if name == "vggish":
        return VGGishBinary(vggish_ckpt=vggish_ckpt)
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
    p.add_argument("--vggish-ckpt", type=str, default=None,
                   help="Path to vggish_model.ckpt (download from "
                        "https://storage.googleapis.com/audioset/vggish_model.ckpt). "
                        "Required when --model vggish to avoid vanishing gradients.")
    p.add_argument("--max-per-class", type=int, default=None,
                   help="Cap each class at this many samples for a balanced dataset "
                        "(e.g. 3457 to match the minority class size).")
    args = p.parse_args()

    seed_everything(args.seed)

    cfg = SonicsConfig(
        root=args.data_root,
        clip_seconds=args.clip_seconds,
        real_subdir=args.real_subdir,
        fake_subdir=args.fake_subdir,
        max_per_class=args.max_per_class,
    )
    full = SonicsDataset(cfg)
    train_set, val_set, class_weights = stratified_split(full, args.val_frac)
    # balanced dataset → uniform weights; imbalanced → inverse-frequency weights
    print(f"Dataset: {len(full)} total  →  {len(train_set)} train / {len(val_set)} val")
    print(f"Class weights (inverse-freq): {class_weights.tolist()}")

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

    print(f"DataLoaders: {len(train_loader)} train batches  →  {len(val_loader)} val batches")

    model = build_model(args.model, vggish_ckpt=args.vggish_ckpt)
    lit = RealFakeLitModule(model, lr=args.lr, class_weights=class_weights)

    logger = TensorBoardLogger(save_dir=str(args.out_dir), name=args.model)
    ckpt = ModelCheckpoint(monitor="val/eer", mode="min", save_top_k=2)
    ckpt_periodic = ModelCheckpoint(
        every_n_epochs=10,
        save_top_k=-1,
        filename="epoch-{epoch:03d}",
    )

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
        callbacks=[ckpt, ckpt_periodic],
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