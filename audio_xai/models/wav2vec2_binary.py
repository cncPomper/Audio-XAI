"""Wav2Vec2-based backbone for binary audio classification.

Backbone: facebook/wav2vec2-base (pretrained on LibriSpeech 960h).
Replace WAV2VEC2_CHECKPOINT with any Wav2Vec2-compatible HuggingFace model ID
(e.g. "facebook/wav2vec2-large-xlsr-53", "facebook/wav2vec2-base-960h",
 or a locally fine-tuned checkpoint).

Architecture differences from AST:
  - Processes raw waveforms via a CNN feature encoder then Transformer layers.
  - No mel-spectrogram: the differentiable feature transform is the CNN encoder
    built into Wav2Vec2 itself.
  - Grad-CAM target: last Wav2Vec2 Transformer encoder layer's LayerNorm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from .base import AudioClassifier

WAV2VEC2_CHECKPOINT = "facebook/wav2vec2-base"
WAV2VEC2_SAMPLE_RATE = 16_000
WAV2VEC2_HIDDEN_SIZE = 768  # matches wav2vec2-base; update if you swap backbone


class Wav2Vec2Binary(AudioClassifier):
    """Wav2Vec2-based binary classifier wrapping AudioClassifier interface."""

    def __init__(self, pretrained: bool = True):
        super().__init__()

        if pretrained:
            self.backbone = Wav2Vec2Model.from_pretrained(WAV2VEC2_CHECKPOINT)
        else:
            cfg = Wav2Vec2Config()
            self.backbone = Wav2Vec2Model(cfg)

        self.classifier = nn.Linear(WAV2VEC2_HIDDEN_SIZE, 2)

    # ------------------------------------------------------------------
    # AudioClassifier interface
    # ------------------------------------------------------------------

    def waveform_to_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pass raw waveform through the Wav2Vec2 CNN feature encoder.

        Parameters:
            waveform: [B, T] at 16 kHz.

        Returns:
            [B, T', hidden_size] — Transformer-ready feature sequence.
        """
        features = self.backbone.feature_extractor(waveform)  # [B, C, T']
        features = features.transpose(1, 2)                    # [B, T', C]
        features, _ = self.backbone.feature_projection(features)
        return features

    def features_to_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Run Transformer layers then pool to logits.

        Parameters:
            features: [B, T', hidden_size].

        Returns:
            [B, 2] logits.
        """
        out = self.backbone.encoder(features).last_hidden_state  # [B, T', H]
        pooled = out.mean(dim=1)                                  # [B, H]
        return self.classifier(pooled)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.waveform_to_features(waveform)
        return self.features_to_logits(features)

    @property
    def target_layer(self) -> nn.Module:
        # Last Transformer encoder layer's final LayerNorm — analogous to AST.
        layer = self.backbone.encoder.layers[-1].final_layer_norm
        assert isinstance(layer, nn.Module)
        return layer