"""Audio Spectrogram Transformer wrapper for binary real/fake classification.

We deliberately do NOT use ``ASTFeatureExtractor`` at training/attack time
because it operates on numpy and breaks gradient flow. Instead we replicate
its log-mel + normalization step in pure torch ops. The HF extractor is fine
for one-off inference but useless inside an adversarial optimization loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T
from transformers import ASTConfig, ASTForAudioClassification

from .base import AudioClassifier

# AST defaults from the MIT/ast-finetuned-audioset checkpoint.
AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SAMPLE_RATE = 16_000
AST_N_MELS = 128
AST_N_FFT = 400
AST_HOP = 160
AST_TARGET_FRAMES = 1024  # AST expects ~10.24s at 16kHz / hop 160
AST_MEAN = -4.2677393
AST_STD = 4.5689974


class ASTBinary(AudioClassifier):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        if pretrained:
            self.backbone = ASTForAudioClassification.from_pretrained(
                AST_CHECKPOINT,
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
        else:
            cfg = ASTConfig(num_labels=2)
            self.backbone = ASTForAudioClassification(cfg)

        # Differentiable mel spectrogram. kept on the same device as the model.
        self.mel = T.MelSpectrogram(
            sample_rate=AST_SAMPLE_RATE,
            n_fft=AST_N_FFT,
            hop_length=AST_HOP,
            n_mels=AST_N_MELS,
            f_min=0,
            f_max=AST_SAMPLE_RATE // 2,
            power=2.0,
        )
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def waveform_to_features(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, T] in [-1, 1]
        spec = self.mel(waveform)  # [B, n_mels, frames]
        spec = self.amp_to_db(spec)  # log-mel
        spec = (spec - AST_MEAN) / (AST_STD * 2)

        # Pad/crop time axis to AST_TARGET_FRAMES.
        frames = spec.shape[-1]
        if frames < AST_TARGET_FRAMES:
            spec = nn.functional.pad(spec, (0, AST_TARGET_FRAMES - frames))
        else:
            spec = spec[..., :AST_TARGET_FRAMES]

        # AST wants [B, frames, n_mels].
        return spec.transpose(-1, -2)

    def features_to_logits(self, features: torch.Tensor) -> torch.Tensor:
        # AST forward signature is (input_values=...). It returns a SequenceClassifierOutput.
        return self.backbone(input_values=features).logits

    @property
    def target_layer(self) -> nn.Module:
        # Last transformer block's layernorm-before-attention works well for
        # attention-rollout-style Grad-CAM. Adjust if you prefer a different layer.
        return self.backbone.audio_spectrogram_transformer.encoder.layer[-1].layernorm_before
