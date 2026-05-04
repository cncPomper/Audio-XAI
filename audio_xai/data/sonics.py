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
    Compute the equal error rate (EER) from model scores and binary labels.
    
    Parameters:
        scores (torch.Tensor): 1-D tensor of model scores or probabilities for each sample.
        labels (torch.Tensor): 1-D tensor of binary labels (0 for negative, 1 for positive) aligned with `scores`.
    
    Returns:
        float: EER expressed as the average of the false positive rate (FPR) and false negative rate (FNR)
        at the score threshold where |FPR - FNR| is minimized. If no positive (or no negative) samples
        are present, the corresponding error rate (FNR or FPR) is treated as 0.0 for computation.
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
        Initialize the Lightning module for binary real/fake classification.
        
        Parameters:
            model (AudioClassifier): Backbone classifier that maps input waveforms to logits for two classes (real vs. fake).
            lr (float): AdamW learning rate.
            weight_decay (float): AdamW weight decay.
        
        Attributes:
            model (AudioClassifier): Stored classifier.
            lr (float): Learning rate used for optimizer.
            weight_decay (float): Weight decay used for optimizer.
            criterion (nn.Module): Cross-entropy loss used for training.
            train_acc (BinaryAccuracy): Training accuracy metric.
            val_acc (BinaryAccuracy): Validation accuracy metric.
            val_auroc (BinaryAUROC): Validation AUROC metric (uses class-1 probability).
            _val_scores (list[torch.Tensor]): Collected class-1 probabilities across validation batches for EER computation.
            _val_labels (list[torch.Tensor]): Collected labels across validation batches for EER computation.
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
        Prepare model forward pass and common metrics for a training/validation step.
        
        Parameters:
            batch (tuple[torch.Tensor, torch.Tensor]): A pair (wav, label) where `wav` is the input tensor for the model and `label` contains integer class targets.
        
        Returns:
            tuple: A 4-tuple containing:
                - loss (torch.Tensor): Cross-entropy loss computed from model logits and `label`.
                - probs_fake (torch.Tensor): Probability of class index 1 for each example (softmax over logits).
                - preds (torch.Tensor): Predicted class indices (argmax over logits).
                - label (torch.Tensor): The original target labels from `batch`.
        """
        wav, label = batch
        logits = self.model(wav)
        loss = self.criterion(logits, label)
        probs_fake = logits.softmax(dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        return loss, probs_fake, preds, label

    def training_step(self, batch, _):
        """
        Performs a single training iteration: computes loss and predictions, updates training accuracy, and logs training loss and accuracy.
        
        Parameters:
            batch: A training batch, expected to contain (waveform, label) as accepted by the module's model.
        
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
        Perform a validation iteration: compute loss, update metrics, accumulate scores/labels, and log validation metrics.
        
        Parameters:
        	batch (tuple): A validation batch, expected to contain input audio and labels (wav, label).
        	_ (Any): Placeholder unused by the method.
        
        Notes:
        	- Updates `val_acc` and `val_auroc` with the batch predictions and probabilities.
        	- Appends per-batch predicted probabilities for the "fake" class to `_val_scores` and labels to `_val_labels`.
        	- Logs "val/loss", "val/acc" (epoch-level), and "val/auroc" (epoch-level).
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
        Compute and log the validation Equal Error Rate (EER) from accumulated batch outputs and clear the accumulators.
        
        If any validation scores were collected, concatenates the stored score and label tensors, computes the EER via `equal_error_rate`, logs it under "val/eer" with `prog_bar=True`, and then clears the internal `_val_scores` and `_val_labels` lists.
        """
        if self._val_scores:
            scores = torch.cat(self._val_scores)
            labels = torch.cat(self._val_labels)
            self.log("val/eer", equal_error_rate(scores, labels), prog_bar=True)
            self._val_scores.clear()
            self._val_labels.clear()

    def configure_optimizers(self):
        """
        Create an AdamW optimizer configured with the module's parameters, learning rate, and weight decay.
        
        Returns:
            torch.optim.Optimizer: AdamW optimizer using self.parameters() with lr=self.lr and weight_decay=self.weight_decay.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
