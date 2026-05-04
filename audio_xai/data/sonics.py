"""LightningModule for binary real/fake training.

Reports the standard deepfake-audio metrics: accuracy, AUROC, EER. EER is the
convention you'll want in any paper/comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from audio_xai.models.base import AudioClassifier


def equal_error_rate(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """EER: threshold where FPR == FNR. Lower is better."""
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()
    thresholds = torch.linspace(scores.min(), scores.max(), steps=1000)
    fars, frrs = [], []
    pos = labels == 1
    neg = labels == 0
    for t in thresholds:
        far = ((scores[neg] >= t).float().mean()).item() if neg.any() else 0.0
        frr = ((scores[pos] < t).float().mean()).item() if pos.any() else 0.0
        fars.append(far)
        frrs.append(frr)
    fars_t = torch.tensor(fars)
    frrs_t = torch.tensor(frrs)
    idx = (fars_t - frrs_t).abs().argmin()
    return ((fars_t[idx] + frrs_t[idx]) / 2).item()


class RealFakeLitModule(LightningModule):
    def __init__(self, model: AudioClassifier, lr: float = 1e-4, weight_decay: float = 1e-5):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()

        self._val_scores: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def _step(self, batch):
        wav, label = batch
        logits = self.model(wav)
        loss = self.criterion(logits, label)
        probs_fake = logits.softmax(dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        return loss, probs_fake, preds, label

    def training_step(self, batch, _):
        loss, _, preds, label = self._step(batch)
        self.train_acc.update(preds, label)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, probs_fake, preds, label = self._step(batch)
        self.val_acc.update(preds, label)
        self.val_auroc.update(probs_fake, label)
        self._val_scores.append(probs_fake)
        self._val_labels.append(label)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("val/auroc", self.val_auroc, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        if self._val_scores:
            scores = torch.cat(self._val_scores)
            labels = torch.cat(self._val_labels)
            self.log("val/eer", equal_error_rate(scores, labels), prog_bar=True)
            self._val_scores.clear()
            self._val_labels.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
