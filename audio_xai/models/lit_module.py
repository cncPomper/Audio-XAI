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
    """
    Compute the Equal Error Rate (EER) for binary classification scores and labels.
    
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
    def __init__(self, model: AudioClassifier, lr: float = 1e-4, weight_decay: float = 1e-5):
        """
        Initialize the Lightning module for binary real/fake audio classification.
        
        Configures the wrapped audio classifier, optimizer hyperparameters, loss criterion, evaluation metrics, and per-epoch buffers for accumulating validation scores and labels used to compute EER.
        
        Parameters:
            model (AudioClassifier): The underlying audio classification model to train and evaluate.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay (L2 regularization) for the optimizer.
        """
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
        """
        Perform a forward pass for a single batch and extract training/metric outputs.
        
        Parameters:
            batch (tuple[torch.Tensor, torch.Tensor]): A pair (wav, label) where `wav` is the input audio tensor and `label` contains integer class labels.
        
        Returns:
            tuple: A 4-tuple containing:
                - loss (torch.Tensor): Cross-entropy loss for the batch.
                - probs_fake (torch.Tensor): Predicted probability for the "fake" class (class index 1) for each sample.
                - preds (torch.Tensor): Predicted class indices for each sample.
                - label (torch.Tensor): The ground-truth labels from the input batch.
        """
        wav, label = batch
        logits = self.model(wav)
        loss = self.criterion(logits, label)
        probs_fake = logits.softmax(dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        return loss, probs_fake, preds, label

    def training_step(self, batch, _):
        """
        Perform a single training iteration: run the model on `batch`, update epoch-level training accuracy, log training loss and accuracy, and return the loss for optimization.
        
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
        """
        Perform validation for a single batch: compute loss and predictions, update validation metrics, buffer scores/labels for epoch-level EER, and log validation metrics.
        
        Parameters:
            batch (tuple): A validation batch, expected as (wav, label) where `wav` is the input audio tensor and `label` contains integer class labels.
        
        Notes:
            - Buffers `self._val_scores` and `self._val_labels` are appended for later EER computation in on_validation_epoch_end.
            - Logs `"val/loss"` (current step), and epoch-aggregated `"val/acc"` and `"val/auroc"`.
        """
        loss, probs_fake, preds, label = self._step(batch)
        self.val_acc.update(preds, label)
        self.val_auroc.update(probs_fake, label)
        self._val_scores.append(probs_fake)
        self._val_labels.append(label)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("val/auroc", self.val_auroc, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        """
        Compute and log the validation Equal Error Rate (EER) from accumulated batch outputs and clear the stored buffers.
        
        If validation scores and labels have been collected during the epoch, concatenates them, computes EER via equal_error_rate(scores, labels), logs the value under "val/eer" (shown in the progress bar), and then clears the internal buffers used to accumulate scores and labels.
        """
        if self._val_scores:
            scores = torch.cat(self._val_scores)
            labels = torch.cat(self._val_labels)
            self.log("val/eer", equal_error_rate(scores, labels), prog_bar=True)
            self._val_scores.clear()
            self._val_labels.clear()

    def configure_optimizers(self):
        """
        Create an AdamW optimizer configured for this module's parameters.
        
        Returns:
            optimizer: An instance of torch.optim.AdamW using this module's parameters with the configured learning rate (`self.lr`) and weight decay (`self.weight_decay`).
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
