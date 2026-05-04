"""VGGish wrapper for binary real/fake classification.

VGGish is the CNN baseline. We use torchvggish-style architecture loaded from torch.hub. Grad-CAM was designed for exactly this kind of model — the
last conv block gives clean, localized attribution maps over the spectrogram.

Note: VGGish operates on 96-frame log-mel patches at 16 kHz (~0.96 s each). For long clips you average logits over patches.
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

    Faithful to the original architecture. If you want pretrained weights, load the harritaylor/torchvggish checkpoint and copy state_dict here.
    """

    def __init__(self):
        """Constructs the VGGish-like convolutional feature extractor and its 128‑dimensional embedding head.

        Initializes two modules on the instance:
        - self.features: convolutional backbone mapping spectrograms to 512-channel feature maps.
        - self.embedding: head flattening features and projecting to 128-D embeddings via FC layers.

        Notes:
        - The embedding linear layer sizes assume the backbone output
            spatial resolution is 6 × 4 (e.g., an input patch that yields 96 time frames and 64 mel bins after four pooling operations).
        """
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
        """Compute a 128-dimensional embedding from a single spectrogram patch.

        Parameters:
            x (torch.Tensor): Input spectrogram patch tensor with shape [B, 1, n_mels, F], where B is batch size,
                n_mels is number of mel bins (64) and F is frames per patch (96).

        Returns:
            torch.Tensor: Embedding tensor with shape [B, 128].
        """
        feats = self.features(x)
        return self.embedding(feats)


class VGGishBinary(AudioClassifier):
    def __init__(self):
        """Initialize the VGGishBinary audio classifier by creating the feature backbone, classification head, and fixed mel-spectrogram
        preprocessing.

        Initializes:
        - a minimal VGGish-like backbone that produces 128-D embeddings,
        - a linear classification head mapping embeddings to 2 logits (real/fake),
        - a MelSpectrogram frontend configured for 16 kHz audio, 64 mel bins, FFT=400, hop=160, and frequency range 125–7500 Hz,
        - an AmplitudeToDB transform for power-to-decibel normalization with a 80 dB top limit.
        """
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
        """Convert a batch of raw waveforms into log-mel spectrogram patches sized for the VGGish backbone.

        Parameters:
            waveform (torch.Tensor): Waveforms with shape [B, T], sampled at VGGISH_SAMPLE_RATE.
                Each batch element is converted to a mel spectrogram and normalized to decibels.

        Returns:
            torch.Tensor: Tensor of log-mel spectrogram patches with shape [B, 1, VGGISH_N_MELS, VGGISH_FRAMES_PER_PATCH].
                Spectrograms are padded or truncated on the time axis so each patch has exactly VGGISH_FRAMES_PER_PATCH frames.
        """
        spec = self.mel(waveform)  # [B, n_mels, frames]
        spec = self.amp_to_db(spec)
        # Take the first 96-frame window for simplicity; for production,
        # tile into multiple patches and average logits.
        F = VGGISH_FRAMES_PER_PATCH

        frames = spec.shape[-1]

        # Pad end so length is an exact multiple of F (never truncates).
        remainder = frames % F
        if remainder != 0:
            spec = nn.functional.pad(spec, (0, F - remainder))

        num_patches = spec.shape[-1] // F
        B, n_mels, _ = spec.shape

        # Reshape: [B, n_mels, num_patches * F] -> [B, num_patches, 1, n_mels, F]
        spec = spec.reshape(B, n_mels, num_patches, F)  # [B, n_mels, P, F]
        spec = spec.permute(0, 2, 1, 3)  # [B, P, n_mels, F]
        return spec.unsqueeze(2)  # [B, P, 1, n_mels, F]

    def features_to_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Convert a batch of spectrogram patch tensors into 2-class logits using the backbone embedding and linear head.

        Parameters:
                features (torch.Tensor): Spectrogram patches shaped [B, 1, n_mels, F], where F is the frames-per-patch (typically 96).

        Returns:
                logits (torch.Tensor): Unnormalized class scores with shape [B, 2].
        """
        if features.dim() == 5:
            B, P, C, n_mels, F = features.shape
            # Flatten batch and patch dims, process all patches in one forward pass.
            flat = features.reshape(B * P, C, n_mels, F)  # [B*P, 1, n_mels, F]
            emb = self.backbone(flat)  # [B*P, 128]
            logits = self.head(emb)  # [B*P, 2]
            logits = logits.reshape(B, P, 2)  # [B, P, 2]
            return logits.mean(dim=1)  # [B, 2]
        else:
            emb = self.backbone(features)
            return self.head(emb)

    @property
    def target_layer(self) -> nn.Module:
        # Last conv layer — canonical Grad-CAM target.
        """Get the backbone's final convolutional layer used as the canonical Grad-CAM target.

        Returns:
            nn.Module: The last convolutional convolutional layer from `self.backbone.features`, intended for Grad-CAM attribution.
        """
        return self.backbone.features[-3]
