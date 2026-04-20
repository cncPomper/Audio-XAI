"""Vision Transformer (ViT) adapted for audio spectrogram classification."""

import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class AudioViT(nn.Module):
    """Vision Transformer (ViT-B/16) adapted for audio spectrogram classification.

    The first convolutional projection layer is replaced so that the model
    accepts single-channel (mono) mel-spectrograms instead of three-channel
    RGB images.  The classification head is replaced with a linear layer
    that outputs ``num_classes`` logits.

    Args:
        num_classes: Number of output classes. Defaults to 527 (AudioSet).
        pretrained: Whether to initialise from ImageNet-21k weights.
            Defaults to ``False``.
    """

    def __init__(self, num_classes: int = 527, pretrained: bool = False) -> None:
        super().__init__()

        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = vit_b_16(weights=weights)

        # Replace patch embedding projection: 3-channel → 1-channel
        original_proj = self.vit.conv_proj
        self.vit.conv_proj = nn.Conv2d(
            1,
            original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
        )

        # Replace classification head
        in_features: int = self.vit.heads.head.in_features  # type: ignore[union-attr]
        self.vit.heads = nn.Sequential(nn.Linear(in_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input spectrogram tensor of shape ``(batch, 1, height, width)``.
                The spatial size must be compatible with the 16×16 patch stride
                (e.g. 224×224).

        Returns:
            Class logits of shape ``(batch, num_classes)``.
        """
        return self.vit(x)
