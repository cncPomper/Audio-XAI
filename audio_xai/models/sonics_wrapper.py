"""Adapter that makes HFAudioClassifier usable as an AudioClassifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_xai.models.base import AudioClassifier


class SonicsWrapper(AudioClassifier):
    """Wraps Sonics HFAudioClassifier to satisfy the AudioClassifier interface."""

    def __init__(self, sonics_model) -> None:
        nn.Module.__init__(self)
        self._m = sonics_model

    def waveform_to_features(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self._m.ft_extractor(waveform)
        spec = spec.unsqueeze(1)
        return F.interpolate(
            spec, size=tuple(self._m.input_shape), mode="bilinear", align_corners=False
        )

    def features_to_logits(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self._m.encoder(features)
        embeds = tokens.mean(dim=1)
        return self._m.classifier(embeds)

    @property
    def target_layer(self) -> nn.Module:
        return self._m.encoder.transformer.blocks[-1]
