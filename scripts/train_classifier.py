"""Minimal training entry point.

Usage:
    python -m audio_xai.scripts.train_classifier \\
        --data-root /mnt/sonics \\
        --model ast \\
        --batch-size 8 \\
        --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
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
    raise ValueError(f"Unknown model: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model", choices=["ast", "vggish"], default="ast")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip-seconds", type=float, default=10.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = p.parse_args()

    seed_everything(args.seed)

    cfg = SonicsConfig(root=args.data_root, clip_seconds=args.clip_seconds)
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

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[ckpt],
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
    )
    trainer.fit(lit, train_loader, val_loader)


if __name__ == "__main__":
    main()
