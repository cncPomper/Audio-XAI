"""Audio Spectrogram Transformer wrapper for binary real/fake classification.

We deliberately do NOT use ``ASTFeatureExtractor`` at training/attack time because it operates on numpy and breaks gradient flow. Instead we replicate
its log-mel + normalization step in pure torch ops. The HF extractor is fine for one-off inference but useless inside an adversarial optimization
loop.
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
        """Constructs an ASTBinary instance with an AST backbone and Torch-based differentiable audio preprocessing.

        When `pretrained=True`, loads the Hugging Face AST checkpoint configured for 2 labels (allowing mismatched sizes).
        When `pretrained=False`, instantiates a fresh AST model configuration for 2 labels. Also sets up differentiable, device-local preprocessing modules:
        - a MelSpectrogram configured with the module-level AST constants (sample rate, n_fft, hop_length, n_mels, f_min=0, f_max=sample_rate/2, power=2.0) and an AmplitudeToDB transform (stype="power", top_db=80) for log-mel conversion.

        Parameters:
            pretrained (bool): If true, load pretrained AST weights; otherwise initialize a new AST model.
        """
        super().__init__()

        if pretrained:
            self.backbone = ASTForAudioClassification.from_pretrained(
                AST_CHECKPOINT,
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
        else:
            cfg = ASTConfig()
            cfg.num_labels = 2
            self.backbone = ASTForAudioClassification(cfg)

        # Differentiable mel spectrogram. kept on the same device as the model.

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
        """Convert a batch of waveforms into AST-compatible, normalized log-mel feature frames.

        Parameters:
            waveform (torch.Tensor): Batch of audio waveforms with shape [B, T]; samples expected in the range [-1, 1].

        Returns:
            torch.Tensor: Log-mel features shaped [B, AST_TARGET_FRAMES, AST_N_MELS], normalized using AST_MEAN and AST_STD. The time axis is padded or trimmed to exactly AST_TARGET_FRAMES.
        """
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
        """Convert precomputed AST features into class logits.

        Parameters:
            features (torch.Tensor): Input features with shape [B, frames, n_mels], matching the AST backbone's expected layout and dtype.

        Returns:
            torch.Tensor: Raw, unnormalized logits of shape [B, 2], where each row contains scores for the two binary classes.
        """
        return self.backbone(input_values=features).logits

    @property
    def target_layer(self) -> nn.Module:
        layer = self.backbone.audio_spectrogram_transformer.encoder.layer[-1].layernorm_before
        assert isinstance(layer, nn.Module)
        return layer
