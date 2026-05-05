"""LightningModule for binary real/fake training.

Reports the standard deepfake-audio metrics: accuracy, AUROC, EER. EER is the convention you'll want in any paper/comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryRecall,
    BinarySpecificity,
)

from audio_xai.models.base import AudioClassifier


def equal_error_rate(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the Equal Error Rate (EER) for binary classification scores and labels.

    Parameters:
        scores (torch.Tensor): Per-sample model scores for the positive class.
        labels (torch.Tensor): Binary ground-truth labels (0 for negative, 1 for positive).

    Returns:
        float: EER value between 0 and 1 — the average of the false acceptance rate (FAR)
        and false rejection rate (FRR) at the threshold where |FAR − FRR| is minimized.
    """
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
    def __init__(
        self,
        model: AudioClassifier,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        # register_buffer moves class_weights to the correct device automatically
        self.register_buffer("_class_weights", class_weights)

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()
        self.val_sensitivity = BinaryRecall()       # TPR: fake correctly detected
        self.val_specificity = BinarySpecificity()  # TNR: real correctly rejected

        self._val_scores: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def _step(self, batch):
        """Perform a forward pass for a single batch and extract training/metric outputs.

        Parameters:
            batch (tuple[torch.Tensor, torch.Tensor]): Batch pair (wav, label); wav is audio, label is class indices.

        Returns:
            tuple: A 4-tuple containing:
                - loss (torch.Tensor): Cross-entropy loss for the batch.
                - probs_fake (torch.Tensor): Predicted probability for the "fake" class (class index 1) for each sample.
                - preds (torch.Tensor): Predicted class indices for each sample.
                - label (torch.Tensor): The ground-truth labels from the input batch.
        """
        wav, label = batch
        logits = self.model(wav)
        loss = F.cross_entropy(logits, label, weight=self._class_weights)
        probs_fake = logits.softmax(dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        return loss, probs_fake, preds, label

    def training_step(self, batch, _):
        """Perform a single training iteration: run the model on `batch`, update epoch-level training accuracy, log training loss and accuracy, and
        return the loss for optimization.

        Parameters:
            batch (tuple): A training batch typically containing (waveform, label).

        Returns:
            loss (torch.Tensor): The computed training loss for the batch.
        """
        loss, _, preds, label = self._step(batch)
        self.train_acc.update(preds, label)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        """Perform validation for a single batch: compute loss and predictions, update validation metrics, buffer scores/labels for epoch-level EER,
        and log validation metrics.

        Parameters:
            batch (tuple): Validation batch as (wav, label); wav is audio, label contains class indices.

        Notes:
            - Buffers `self._val_scores` and `self._val_labels` are appended for later EER computation in on_validation_epoch_end.
            - Logs `"val/loss"` (current step), and epoch-aggregated `"val/acc"` and `"val/auroc"`.
        """
        loss, probs_fake, preds, label = self._step(batch)
        self.val_acc.update(preds, label)
        self.val_auroc.update(probs_fake, label)
        self.val_f1.update(preds, label)
        self.val_sensitivity.update(preds, label)
        self.val_specificity.update(preds, label)
        self._val_scores.append(probs_fake)
        self._val_labels.append(label)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_epoch=True)
        self.log("val/auroc", self.val_auroc, prog_bar=True, on_epoch=True)
        self.log("val/f1", self.val_f1, on_epoch=True)
        self.log("val/sensitivity", self.val_sensitivity, on_epoch=True)
        self.log("val/specificity", self.val_specificity, on_epoch=True)

    def on_validation_epoch_end(self):
        """Compute and log the validation EER gathered across all ranks."""
        if not self._val_scores:
            self._val_scores.clear()
            self._val_labels.clear()
            return

        scores = torch.cat(self._val_scores)  # [N_local]
        labels = torch.cat(self._val_labels)  # [N_local]
        self._val_scores.clear()
        self._val_labels.clear()

        # Debug: Check score distribution
        if self.trainer.is_global_zero:
            unique_scores = scores.unique()
            print(f"[DEBUG] Validation score distribution: min={scores.min():.4f}, max={scores.max():.4f}, unique_values={len(unique_scores)}")
            print(f"[DEBUG] Label distribution: {labels.unique()} counts: {[(labels==i).sum().item() for i in labels.unique()]}")

        # all_gather is a collective — every rank must call it, never gate with is_global_zero.
        # Returns [world_size, N_local] under DDP, or [N_local] on a single device.
        all_scores = self.all_gather(scores)
        all_labels = self.all_gather(labels)

        assert isinstance(all_scores, torch.Tensor)
        assert isinstance(all_labels, torch.Tensor)
        if self.trainer.is_global_zero:
            full_scores = all_scores.reshape(-1)
            full_labels = all_labels.reshape(-1)
            self.log(
                "val/eer",
                equal_error_rate(full_scores, full_labels),
                prog_bar=True,
                rank_zero_only=True,
            )

    def configure_optimizers(self):
        """Create an AdamW optimizer configured for this module's parameters.

        Returns:
            optimizer: AdamW with this module's params, lr=self.lr, weight_decay=self.weight_decay.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
