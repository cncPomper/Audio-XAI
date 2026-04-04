"""Models for Audio XAI.

Three model families are provided, each accepting a mono mel-spectrogram
tensor of shape ``(batch, 1, freq, time)`` and returning class logits:

* :class:`Spectra` – spectral-decomposition CNN from the Sonics paper
  (https://arxiv.org/abs/2507.10447).
* :class:`AudioViT` – Vision Transformer (ViT-B/16) adapted for audio.
* :class:`AudioResNet50` – ResNet50 adapted for audio.
"""

from audio_xai.models.resnet import AudioResNet50
from audio_xai.models.spectra import Spectra
from audio_xai.models.vit import AudioViT

__all__ = ["AudioResNet50", "AudioViT", "Spectra"]
