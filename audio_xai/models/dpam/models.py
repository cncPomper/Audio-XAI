"""PyTorch network architecture for the DPAM perceptual audio metric.

Port of the original TensorFlow 1.14 / Python 3.7 implementation from:
  Manocha et al., "A Differentiable Perceptual Audio Metric Learned from
  Just Noticeable Differences", Interspeech 2020.
  https://github.com/pranaymanocha/PerceptualAudio
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Single conv → norm → leaky-ReLU block used inside LossNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_type: str,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if norm_type == "SBN":
            self.norm: nn.Module = nn.BatchNorm1d(out_channels)
        elif norm_type == "NM":
            self.norm = nn.InstanceNorm1d(out_channels, affine=True)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.norm(self.conv(x)), negative_slope=0.2)


class LossNet(nn.Module):
    """Multi-layer 1-D convolutional feature extractor.

    Mirrors the ``lossnet`` function from the original TF implementation.
    Channels double every ``blk_channels`` layers starting from
    ``base_channels``.  Each layer applies stride-2 downsampling so that
    high-level perceptual features are captured at multiple temporal scales.

    Args:
        n_layers: Total number of convolutional blocks.
        base_channels: Number of output channels in the first block.
        blk_channels: Channels double every this many layers.
        norm_type: Normalisation – ``"SBN"`` (BatchNorm), ``"NM"``
            (InstanceNorm with affine params), or ``"none"`` (no norm).
        kernel_size: Convolution kernel width (must be odd).
    """

    def __init__(
        self,
        n_layers: int = 14,
        base_channels: int = 32,
        blk_channels: int = 5,
        norm_type: str = "SBN",
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = 1
        for i in range(n_layers):
            out_ch = base_channels * (2 ** (i // blk_channels))
            blocks.append(_ConvBlock(in_ch, out_ch, kernel_size, stride=2, norm_type=norm_type))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return intermediate feature maps from every block.

        Args:
            x: Raw waveform tensor of shape ``[N, 1, L]``.

        Returns:
            List of feature tensors, one per convolutional block.
        """
        features: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class DPAMNet(nn.Module):
    """Full DPAM model: feature extractor + trainable per-layer weights.

    The perceptual distance is the sigmoid-weighted sum of per-layer L1
    differences between the feature maps of the reference and degraded
    signals – equivalent to ``featureloss`` in the original codebase.

    Args:
        n_layers: Passed through to :class:`LossNet`.
        base_channels: Passed through to :class:`LossNet`.
        blk_channels: Passed through to :class:`LossNet`.
        norm_type: Passed through to :class:`LossNet`.
        kernel_size: Passed through to :class:`LossNet`.
    """

    def __init__(
        self,
        n_layers: int = 14,
        base_channels: int = 32,
        blk_channels: int = 5,
        norm_type: str = "SBN",
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.feature_net = LossNet(n_layers, base_channels, blk_channels, norm_type, kernel_size)
        # One trainable scalar weight per layer (sigmoid-activated at runtime)
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, ref: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """Compute the DPAM perceptual distance.

        Args:
            ref: Reference waveform ``[N, 1, L]``.
            deg: Degraded waveform ``[N, 1, L]``.

        Returns:
            Scalar distance tensor.
        """
        feats_ref = self.feature_net(ref)
        feats_deg = self.feature_net(deg)
        weights = torch.sigmoid(self.layer_weights)

        dist = torch.zeros(1, device=ref.device, dtype=ref.dtype)
        for w, f_r, f_d in zip(weights, feats_ref, feats_deg, strict=True):
            min_len = min(f_r.shape[-1], f_d.shape[-1])
            dist = dist + w * torch.mean(torch.abs(f_r[..., :min_len] - f_d[..., :min_len]))
        return dist
