"""Spectra model from the Sonics paper (https://arxiv.org/abs/2507.10447)."""

import torch
import torch.nn as nn


class _SpectraBlock(nn.Module):
    """Residual block with separate spectral (frequency) and temporal convolutions."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        # Frequency-axis convolution
        self.freq_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), bias=False
        )
        self.freq_bn = nn.BatchNorm2d(out_channels)
        # Time-axis convolution
        self.time_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False
        )
        self.time_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.freq_bn(self.freq_conv(x)))
        out = self.time_bn(self.time_conv(out))
        return self.relu(out + self.shortcut(x))


def _make_spectra_layer(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
    layers: list[nn.Module] = [_SpectraBlock(in_channels, out_channels, stride)]
    for _ in range(1, num_blocks):
        layers.append(_SpectraBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class Spectra(nn.Module):
    """Spectra model from the Sonics paper for audio classification.

    Processes mel-spectrograms through separate frequency-axis and time-axis
    convolutions before fusing into a shared representation, following the
    spectral decomposition approach described in the Sonics paper
    (https://arxiv.org/abs/2507.10447).

    Args:
        num_classes: Number of output classes. Defaults to 527 (AudioSet).
        in_channels: Number of input channels. Defaults to 1 (mono spectrogram).
    """

    def __init__(self, num_classes: int = 527, in_channels: int = 1) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = _make_spectra_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = _make_spectra_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = _make_spectra_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = _make_spectra_layer(256, 512, num_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input spectrogram tensor of shape ``(batch, channels, freq, time)``.

        Returns:
            Class logits of shape ``(batch, num_classes)``.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
