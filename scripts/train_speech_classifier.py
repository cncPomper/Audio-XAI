#!/usr/bin/env python3
"""
Fine-tune AST, VGGish, or SpecTra on Google Speech Commands.
=============================================================
Replaces the binary classification head with an N-class head and fine-tunes
on the full Speech Commands dataset downloaded from HuggingFace Hub.

The best checkpoint (by val/acc) is saved to:
    {out_dir}/best.ckpt

Usage:
    python scripts/train_speech_classifier.py \\
        --model ast \\
        --version v0.02 \\
        --epochs 20 \\
        --out-dir runs/speech_ast

    python scripts/train_speech_classifier.py \\
        --model vggish \\
        --vggish-ckpt /path/to/vggish_model.ckpt \\
        --epochs 20 \\
        --out-dir runs/speech_vggish

    python scripts/train_speech_classifier.py \\
        --model spectra \\
        --epochs 20 \\
        --out-dir runs/speech_spectra

SLURM:
    sbatch --gres=gpu:1 --mem=32G --time=8:00:00 \\
        --wrap="python scripts/train_speech_classifier.py \\
            --model ast --epochs 20 --out-dir runs/speech_ast"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_xai.data.speech_commands_hf_dataset import (
    SpeechCommandsHFConfig,
    SpeechCommandsHFDataset,
)
from audio_xai.models.ast_binary import ASTBinary
from audio_xai.models.multiclass_lit_module import MultiClassLitModule, _replace_head
from audio_xai.models.wav2vec2_binary import Wav2Vec2Binary
from audio_xai.models.vggish_binary import VGGishBinary


def _build_model(args):
    if args.model == "ast":
        model = ASTBinary(pretrained=True)
    elif args.model == "vggish":
        model = VGGishBinary(vggish_ckpt=args.vggish_ckpt)
    elif args.model == "spectra":
        model = Wav2Vec2Binary(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {args.model!r}")
    return model


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fine-tune audio classifier on Speech Commands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=["ast", "vggish", "spectra"], default="ast")
    p.add_argument("--version", choices=["v0.01", "v0.02"], default="v0.02",
                   help="Speech Commands dataset version")
    p.add_argument("--skip-unknown", action="store_true",
                   help="Filter out _unknown_ class during training")
    p.add_argument("--skip-silence", action="store_true",
                   help="Filter out _silence_ class during training")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=500,
                   help="Linear LR warm-up steps (0 = off)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=Path("runs/speech_classifier"))
    p.add_argument("--vggish-ckpt", type=str, default=None,
                   help="Path to vggish_model.ckpt (optional, improves gradient flow)")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="HuggingFace dataset cache directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--strategy", type=str, default="auto",
                   help="Lightning strategy: 'auto', 'ddp', 'ddp_spawn'")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    ds_cfg_base = dict(
        version=args.version,
        skip_unknown=args.skip_unknown,
        skip_silence=args.skip_silence,
        cache_dir=args.cache_dir,
    )
    train_ds = SpeechCommandsHFDataset(SpeechCommandsHFConfig(split="train",      **ds_cfg_base))
    val_ds   = SpeechCommandsHFDataset(SpeechCommandsHFConfig(split="validation", **ds_cfg_base))

    n_classes = train_ds.n_classes
    print(f"Classes: {n_classes}  —  {train_ds.label_names}")
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = _build_model(args)
    _replace_head(model, n_classes)
    print(f"[model] {args.model} — head replaced → {n_classes} classes")

    # Enable gradient checkpointing for memory-intensive transformers.
    if hasattr(model, "backbone") and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    lit = MultiClassLitModule(
        model=model,
        n_classes=n_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.out_dir,
        filename="best",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=str(args.out_dir), name="tb")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        callbacks=[checkpoint_cb, lr_monitor],
        logger=logger,
        log_every_n_steps=20,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        enable_progress_bar=True,
    )

    trainer.fit(lit, train_loader, val_loader)

    best = checkpoint_cb.best_model_path
    print(f"\nBest checkpoint → {best}")
    print(f"Best val/acc     → {checkpoint_cb.best_model_score:.4f}")
    print(
        f"\nUse this checkpoint in run_speech_experiment.py:\n"
        f"  --checkpoint {best}"
    )


if __name__ == "__main__":
    main()