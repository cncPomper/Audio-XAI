"""VGGish wrapper for binary real/fake classification.

Architecture and preprocessing match the official AudioSet VGGish release exactly:
  https://github.com/tensorflow/models/tree/master/research/audioset/vggish
so the published checkpoint can be loaded for fine-tuning:
  https://storage.googleapis.com/audioset/vggish_model.ckpt

Input convention (matches vggish_slim.py after NHWC → NCHW):
  [B, 1, NUM_FRAMES=96, NUM_BANDS=64]  — time axis first, frequency axis second.

Vanishing-gradient fix: load the pretrained checkpoint via VGGishBinary(vggish_ckpt=...)
instead of training from random weights. The published weights already provide stable
gradient flow; BatchNorm is intentionally absent to stay checkpoint-compatible.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T

from .base import AudioClassifier

# Official VGGish hyperparameters — vggish_params.py
VGGISH_SAMPLE_RATE = 16_000
VGGISH_NUM_FRAMES = 96        # time frames per 0.96 s patch
VGGISH_NUM_BANDS = 64         # mel frequency bands
VGGISH_EMBEDDING_SIZE = 128
VGGISH_N_FFT = 512            # FFT size: next power-of-2 ≥ 25 ms window (400 samples)
VGGISH_WIN_LENGTH = 400       # 25 ms window at 16 kHz
VGGISH_HOP = 160              # 10 ms hop at 16 kHz
VGGISH_MEL_MIN_HZ = 125
VGGISH_MEL_MAX_HZ = 7500
VGGISH_LOG_OFFSET = 0.01      # log(mel + offset) stabilisation — vggish_params.LOG_OFFSET
VGGISH_INIT_STDDEV = 0.01     # truncated-normal stddev — vggish_params.INIT_STDDEV


class _VGGishBackbone(nn.Module):
    """Exact VGGish feature extractor: 4 conv blocks + 3-layer FC embedding head.

    Matches vggish_slim.py precisely — no BatchNorm, no Dropout — so the
    published TF checkpoint loads cleanly via load_vggish_checkpoint().

    Input : [B, 1, NUM_FRAMES=96, NUM_BANDS=64]
    Output: [B, EMBEDDING_SIZE=128]  (pre-activation, no final ReLU)
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — conv1 / pool1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2 — conv2 / pool2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3 — conv3_{1,2} / pool3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4 — conv4_{1,2} / pool4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),   # index -3: Grad-CAM target
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # After 4 × MaxPool2d(2,2) on [96, 64]: spatial → [6, 4], channels → 512
        # fc1 uses slim.repeat(2, ...) with activation_fn=relu; fc2 has activation_fn=None
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 4, 4096),   # fc1_1
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),            # fc1_2
            nn.ReLU(inplace=True),
            nn.Linear(4096, VGGISH_EMBEDDING_SIZE),  # fc2 — no activation
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Truncated-normal init (σ=0.01, clipped at ±2σ) matching vggish_slim.py."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(
                    m.weight,
                    mean=0.0,
                    std=VGGISH_INIT_STDDEV,
                    a=-2 * VGGISH_INIT_STDDEV,
                    b=2 * VGGISH_INIT_STDDEV,
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_vggish_checkpoint(self, ckpt_path: str) -> None:
        """Copy official VGGish weights from a TF1 checkpoint into this module.

        Download: https://storage.googleapis.com/audioset/vggish_model.ckpt
        Requires: pip install tensorflow

        TF→PyTorch conversions applied:
          Conv weights:  [H, W, C_in, C_out] → [C_out, C_in, H, W]  (HWIO→OIHW)
          FC weights:    [in, out]            → [out, in]             (.T)
          fc1_1 weight:  rows permuted to reconcile TF NHWC vs PyTorch NCHW
                         flatten ordering after pool4.
        """
        try:
            import tensorflow.compat.v1 as tf
        except ImportError:
            raise RuntimeError(
                "TensorFlow is required to load the VGGish checkpoint. "
                "Install with: pip install tensorflow"
            )
        reader = tf.train.load_checkpoint(ckpt_path)

        def _t(name: str) -> np.ndarray:
            return reader.get_tensor(name)

        def _conv(name: str) -> torch.Tensor:
            return torch.from_numpy(np.transpose(_t(name), (3, 2, 0, 1)))

        def _bias(name: str) -> torch.Tensor:
            return torch.from_numpy(_t(name))

        def _fc(name: str) -> torch.Tensor:
            return torch.from_numpy(_t(name).T)

        with torch.no_grad():
            # ── Conv layers ────────────────────────────────────────────────────
            self.features[0].weight.copy_(_conv("vggish/conv1/weights"))
            self.features[0].bias.copy_(_bias("vggish/conv1/biases"))

            self.features[3].weight.copy_(_conv("vggish/conv2/weights"))
            self.features[3].bias.copy_(_bias("vggish/conv2/biases"))

            self.features[6].weight.copy_(_conv("vggish/conv3/conv3_1/weights"))
            self.features[6].bias.copy_(_bias("vggish/conv3/conv3_1/biases"))

            self.features[8].weight.copy_(_conv("vggish/conv3/conv3_2/weights"))
            self.features[8].bias.copy_(_bias("vggish/conv3/conv3_2/biases"))

            self.features[11].weight.copy_(_conv("vggish/conv4/conv4_1/weights"))
            self.features[11].bias.copy_(_bias("vggish/conv4/conv4_1/biases"))

            self.features[13].weight.copy_(_conv("vggish/conv4/conv4_2/weights"))
            self.features[13].bias.copy_(_bias("vggish/conv4/conv4_2/biases"))

            # ── fc1_1: reconcile flatten order (NHWC vs NCHW) ─────────────────
            # TF pool4 output: [B, H=6, W=4, C=512]; flatten index = h*(W*C)+w*C+c
            # PT pool4 output: [B, C=512, H=6, W=4]; flatten index = c*(H*W)+h*W+w
            H, W, C = 6, 4, 512
            perm = np.empty(H * W * C, dtype=np.int64)
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        perm[c * (H * W) + h * W + w] = h * (W * C) + w * C + c
            fc1_w = _t("vggish/fc1/fc1_1/weights")           # [12288, 4096]
            self.embedding[1].weight.copy_(torch.from_numpy(fc1_w[perm].T))
            self.embedding[1].bias.copy_(_bias("vggish/fc1/fc1_1/biases"))

            # ── fc1_2, fc2 ────────────────────────────────────────────────────
            self.embedding[3].weight.copy_(_fc("vggish/fc1/fc1_2/weights"))
            self.embedding[3].bias.copy_(_bias("vggish/fc1/fc1_2/biases"))

            self.embedding[5].weight.copy_(_fc("vggish/fc2/weights"))
            self.embedding[5].bias.copy_(_bias("vggish/fc2/biases"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.features(x))


class VGGishBinary(AudioClassifier):
    def __init__(self, vggish_ckpt: str | None = None) -> None:
        """Binary real/fake classifier built on the official VGGish backbone.

        Args:
            vggish_ckpt: Optional path to the official TF checkpoint
                (https://storage.googleapis.com/audioset/vggish_model.ckpt).
                When supplied, backbone weights are initialised from the published
                AudioSet model, which provides stable gradient flow for fine-tuning
                without requiring BatchNorm.
        """
        super().__init__()
        self.backbone = _VGGishBackbone()
        self.head = nn.Linear(VGGISH_EMBEDDING_SIZE, 2)

        # Preprocessing — vggish_input.py: 25 ms window / 10 ms hop, 64 mel bands
        self.mel = T.MelSpectrogram(
            sample_rate=VGGISH_SAMPLE_RATE,
            n_fft=VGGISH_N_FFT,
            win_length=VGGISH_WIN_LENGTH,
            hop_length=VGGISH_HOP,
            n_mels=VGGISH_NUM_BANDS,
            f_min=VGGISH_MEL_MIN_HZ,
            f_max=VGGISH_MEL_MAX_HZ,
            power=2.0,
        )

        if vggish_ckpt is not None:
            self.backbone.load_vggish_checkpoint(vggish_ckpt)

    def waveform_to_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveforms [B, T] → log-mel patches [B, P, 1, NUM_FRAMES=96, NUM_BANDS=64].

        Matches vggish_input.waveform_to_examples: power mel-spectrogram → log(mel + 0.01).
        The time axis is first (dim -2) so the backbone sees [frames × bands] as in the
        original TF model ([H=96, W=64] after NHWC→NCHW transposition).
        Short clips are zero-padded; long clips are split into non-overlapping 0.96 s patches.
        """
        spec = self.mel(waveform)                          # [B, NUM_BANDS=64, T_frames]
        spec = torch.log(spec + VGGISH_LOG_OFFSET)         # log(mel + 0.01)

        B, n_bands, n_frames = spec.shape
        F = VGGISH_NUM_FRAMES
        remainder = n_frames % F
        if remainder != 0:
            spec = nn.functional.pad(spec, (0, F - remainder))

        num_patches = spec.shape[-1] // F
        # [B, 64, P*F] → [B, 64, P, F] → [B, P, F=96, 64] → [B, P, 1, 96, 64]
        spec = spec.reshape(B, n_bands, num_patches, F)
        spec = spec.permute(0, 2, 3, 1)                    # [B, P, 96, 64]
        return spec.unsqueeze(2)                            # [B, P, 1, 96, 64]

    def features_to_logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 5:
            B, P, C, n_frames, n_bands = features.shape
            flat = features.reshape(B * P, C, n_frames, n_bands)
            emb = self.backbone(flat)                       # [B*P, 128]
            return self.head(emb).reshape(B, P, 2).mean(dim=1)
        return self.head(self.backbone(features))

    @property
    def target_layer(self) -> nn.Module:
        # features[-3] = Conv2d(512, 512) — last conv before pool4, canonical Grad-CAM target
        return self.backbone.features[-3]