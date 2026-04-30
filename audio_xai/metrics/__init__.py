"""Audio feature extraction and quality metrics.

Includes a PyTorch port of the Deep Perceptual Audio Metric (DPAM) by
Manocha et al. (Interspeech 2020).
"""

from audio_xai.metrics.dpam import DPAM, load_audio

__all__ = ["DPAM", "load_audio"]
