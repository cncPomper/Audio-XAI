"""ResNet50 adapted for audio spectrogram classification."""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class AudioResNet50(nn.Module):
    """ResNet50 adapted for audio spectrogram classification.

    The first convolutional layer is replaced to accept single-channel
    (mono) mel-spectrograms instead of three-channel RGB images.
    The fully-connected head is replaced to output ``num_classes`` logits.

    Args:
        num_classes: Number of output classes. Defaults to 527 (AudioSet).
        pretrained: Whether to initialise from ImageNet-1k weights.
            Defaults to ``False``.
    """

    def __init__(self, num_classes: int = 527, pretrained: bool = False) -> None:
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = resnet50(weights=weights)

        # Replace first conv: 3-channel RGB → 1-channel mono spectrogram
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Replace classification head
        in_features: int = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input spectrogram tensor of shape ``(batch, 1, height, width)``.

        Returns:
            Class logits of shape ``(batch, num_classes)``.
        """
        return self.resnet(x)
