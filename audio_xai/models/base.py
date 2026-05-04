"""Shared interface for audio classifiers.

The XAI methods (Grad-CAM, IG) and the attack loop should not need to know
whether they're talking to AST, VGGish, or anything else. This base class
defines the contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AudioClassifier(nn.Module, ABC):
    """Binary real/fake audio classifier with a uniform interface.

    Subclasses must implement:
        - ``waveform_to_features``: raw waveform [B, T] -> spectrogram features
          shaped so the backbone can consume them.
        - ``forward``: features -> logits [B, 2].
        - ``target_layer``: the conv/attention layer Grad-CAM hooks into.

    The split is intentional. XAI methods need (a) the spectrogram (the thing
    explanations are computed over) and (b) a hook layer. The attack loop
    needs to differentiate through the spectrogram step, so we keep it inside
    the module.
    """

    n_classes: int = 2

    @abstractmethod
    def waveform_to_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of raw waveforms into spectrogram-like feature tensors for classification.
        
        Parameters:
            waveform (torch.Tensor): Input waveforms with shape [B, T], where B is batch size and T is time samples.
        
        Returns:
            torch.Tensor: Feature tensor with shape [B, C, F, T'] (channels, frequency bins, time frames) suitable for downstream classification.
        
        Notes:
            This transformation must be end-to-end differentiable (no detach or CPU/NumPy round-trips).
        """

    @abstractmethod
    def features_to_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        Map spectrogram-like features to classification logits.
        
        Parameters:
            features (torch.Tensor): Input features with shape [B, C, F, T'].
        
        Returns:
            torch.Tensor: Logits with shape [B, n_classes], where each row contains unnormalized class scores.
        """

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for a batch of input waveforms.
        
        Parameters:
            waveform (torch.Tensor): Input waveform tensor with shape [B, T], where B is batch size and T is number of time samples.
        
        Returns:
            torch.Tensor: Logits with shape [B, n_classes].
        """
        return self.features_to_logits(self.waveform_to_features(waveform))

    @property
    @abstractmethod
    def target_layer(self) -> nn.Module:
        """
        The neural network module where Grad-CAM hooks should be attached.
        
        Returns:
            layer (nn.Module): The module instance to attach Grad-CAM hooks to (provides activations and gradients).
        """
