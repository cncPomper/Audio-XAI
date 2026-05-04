"""VGGish wrapper for binary real/fake classification.

VGGish is the CNN baseline. We use torchvggish-style architecture loaded from
torch.hub. Grad-CAM was designed for exactly this kind of model — the last
conv block gives clean, localized attribution maps over the spectrogram.

Note: VGGish operates on 96-frame log-mel patches at 16 kHz (~0.96 s each).
For long clips you average logits over patches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T

from .base import AudioClassifier

VGGISH_SAMPLE_RATE = 16_000
VGGISH_N_MELS = 64
VGGISH_N_FFT = 400
VGGISH_HOP = 160
VGGISH_FRAMES_PER_PATCH = 96  # ~0.96 s


class _VGGishBackbone(nn.Module):
    """Minimal VGGish feature extractor (4 conv blocks).

    Faithful to the original architecture. If you want pretrained weights,
    load the harritaylor/torchvggish checkpoint and copy state_dict here.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # After 4 pools on 96x64: spatial = 6 x 4 = 24, channels = 512
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.embedding(feats)


class VGGishBinary(AudioClassifier):
    def __init__(self):
        super().__init__()
        self.backbone = _VGGishBackbone()
        self.head = nn.Linear(128, 2)

        self.mel = T.MelSpectrogram(
            sample_rate=VGGISH_SAMPLE_RATE,
            n_fft=VGGISH_N_FFT,
            hop_length=VGGISH_HOP,
            n_mels=VGGISH_N_MELS,
            f_min=125,
            f_max=7500,
            power=2.0,
        )
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def waveform_to_features(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, T]
        spec = self.mel(waveform)  # [B, n_mels, frames]
        spec = self.amp_to_db(spec)
        # Take the first 96-frame window for simplicity; for production,
        # tile into multiple patches and average logits.
        F = VGGISH_FRAMES_PER_PATCH
        if spec.shape[-1] < F:
            spec = nn.functional.pad(spec, (0, F - spec.shape[-1]))
        spec = spec[..., :F]
        # Add channel dim and put time on last axis: [B, 1, n_mels, F].
        return spec.unsqueeze(1)

    def features_to_logits(self, features: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(features)
        return self.head(emb)

    @property
    def target_layer(self) -> nn.Module:
        # Last conv layer — canonical Grad-CAM target.
        return self.backbone.features[-3]
