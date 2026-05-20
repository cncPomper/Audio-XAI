"""Lightning module for multi-class speech command classification.

Drop-in replacement for RealFakeLitModule when training on Speech Commands
(35-class keyword spotting) instead of the binary SONICS real/fake task.

Metrics logged:
  train/loss, train/acc
  val/loss, val/acc, val/top3_acc, val/f1_macro
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)

from audio_xai.models.base import AudioClassifier


def _replace_head(model: AudioClassifier, n_classes: int) -> None:
    """Replace the classification head of any supported model in-place.

    Supported model types:
      - VGGishBinary:  model.head = Linear(128, n_classes)
      - ASTBinary:     model.backbone.classifier.dense = Linear(768, n_classes)
      - Wav2Vec2Binary: model.classifier = Linear(hidden_size, n_classes)
    """
    cls = type(model).__name__

    if cls == "VGGishBinary":
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, n_classes)

    elif cls == "ASTBinary":
        dense = model.backbone.classifier.dense
        model.backbone.classifier.dense = nn.Linear(dense.in_features, n_classes)

    elif cls == "Wav2Vec2Binary":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, n_classes)

    else:
        raise ValueError(
            f"_replace_head: unsupported model type {cls!r}. "
            "Manually replace the head before passing to MultiClassLitModule."
        )


class MultiClassLitModule(LightningModule):
    """PyTorch Lightning module for multi-class audio classification.

    Args:
        model:       An AudioClassifier instance whose head already outputs
                     n_classes logits (use _replace_head if needed).
        n_classes:   Number of output classes.
        lr:          AdamW learning rate.
        weight_decay: AdamW weight decay.
        class_weights: Optional per-class loss weights [n_classes].
        warmup_steps: Linear warm-up duration in optimizer steps (0 = off).
    """

    def __init__(
        self,
        model: AudioClassifier,
        n_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        class_weights: torch.Tensor | None = None,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.model = model
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.register_buffer("_class_weights", class_weights)

        self.train_acc   = MulticlassAccuracy(n_classes, average="micro")
        self.val_acc     = MulticlassAccuracy(n_classes, average="micro")
        self.val_top3    = MulticlassAccuracy(n_classes, average="micro", top_k=3)
        self.val_f1      = MulticlassF1Score(n_classes, average="macro")

    # ------------------------------------------------------------------

    def _step(self, batch):
        wav, label = batch
        logits = self.model(wav)
        loss = F.cross_entropy(logits, label, weight=self._class_weights)
        preds = logits.argmax(dim=-1)
        return loss, preds, label

    def training_step(self, batch, _):
        loss, preds, label = self._step(batch)
        self.train_acc.update(preds, label)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc",  self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, preds, label = self._step(batch)
        self.val_acc.update(preds, label)
        self.val_top3.update(self.model(batch[0]).softmax(-1), label)
        self.val_f1.update(preds, label)
        self.log("val/loss",      loss,           prog_bar=True)
        self.log("val/acc",       self.val_acc,   prog_bar=True, on_epoch=True)
        self.log("val/top3_acc",  self.val_top3,  on_epoch=True)
        self.log("val/f1_macro",  self.val_f1,    on_epoch=True)

    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.warmup_steps,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer