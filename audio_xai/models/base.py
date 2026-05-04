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
        """[B, T] waveform -> [B, C, F, T'] spectrogram-like features.

        Must be differentiable end-to-end (no detach, no numpy roundtrip).
        """

    @abstractmethod
    def features_to_logits(self, features: torch.Tensor) -> torch.Tensor:
        """[B, C, F, T'] features -> [B, n_classes] logits."""

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.features_to_logits(self.waveform_to_features(waveform))

    @property
    @abstractmethod
    def target_layer(self) -> nn.Module:
        """Layer used as the Grad-CAM hook point."""
