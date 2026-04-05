"""Spectra model from https://arxiv.org/abs/2408.14080v3."""

from __future__ import annotations

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
    """Spectra model for audio spectrogram classification.

    Paper: https://arxiv.org/abs/2408.14080v3

    Can operate in two modes controlled by ``pretrained``:

    * **pretrained=False** (default): Uses a custom spectral-decomposition CNN
      with separate frequency-axis ``(3×1)`` and time-axis ``(1×3)`` residual
      convolutions.  Accepts tensors of shape
      ``(batch, in_channels, freq, time)``.

    * **pretrained=True**: Loads a HuggingFace Hub checkpoint via
      ``transformers.AutoModelForAudioClassification``.  The ``forward()``
      method passes ``pixel_values`` to the upstream model and returns logits,
      so callers should prepare inputs using the corresponding feature
      extractor / processor.  Requires the ``transformers`` package.

    Args:
        num_classes: Number of output classes. Defaults to 527 (AudioSet).
        in_channels: Number of input channels for the custom architecture.
            Ignored when ``pretrained=True``.
        pretrained: Load a pretrained checkpoint from HuggingFace Hub.
            Requires ``hf_model_name`` and the ``transformers`` package.
            Defaults to ``False``.
        hf_model_name: HuggingFace Hub model identifier used when
            ``pretrained=True``.  Find the official checkpoint published
            alongside https://arxiv.org/abs/2408.14080v3 on HuggingFace Hub.
    """

    def __init__(
        self,
        num_classes: int = 527,
        in_channels: int = 1,
        pretrained: bool = False,
        hf_model_name: str | None = None,
    ) -> None:
        super().__init__()
        self._pretrained = pretrained

        if pretrained:
            if hf_model_name is None:
                raise ValueError(
                    "hf_model_name must be provided when pretrained=True. "
                    "Supply the HuggingFace model ID from https://arxiv.org/abs/2408.14080v3."
                )
            try:
                from transformers import AutoModelForAudioClassification
            except ImportError as exc:
                raise ImportError(
                    "The 'transformers' package is required for pretrained=True. "
                    "Install it with: pip install 'Audio-XAI[models]'"
                ) from exc
            self._hf_model: nn.Module | None = AutoModelForAudioClassification.from_pretrained(
                hf_model_name, num_labels=num_classes, ignore_mismatched_sizes=True
            )
        else:
            self._hf_model = None

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

        When ``pretrained=True`` the input ``x`` is forwarded as
        ``pixel_values`` to the underlying HuggingFace model; prepare it with
        the corresponding feature extractor / processor.

        When ``pretrained=False`` ``x`` should have shape
        ``(batch, in_channels, freq, time)``.

        Args:
            x: Input tensor.

        Returns:
            Class logits of shape ``(batch, num_classes)``.
        """
        if self._pretrained:
            outputs = self._hf_model(pixel_values=x)  # type: ignore[call-arg]
            return outputs.logits

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
