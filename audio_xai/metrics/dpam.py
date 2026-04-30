"""DPAM – Deep Perceptual Audio Metric.

PyTorch / Python >=3.12 port of the original TensorFlow 1.14 implementation:
  https://github.com/pranaymanocha/PerceptualAudio

Usage (mirrors the original pip API)::

    import audio_xai.dpam as dpam

    loss_fn = dpam.DPAM()
    wav_ref = dpam.load_audio("sample_audio/ref.wav")
    wav_out = dpam.load_audio("sample_audio/2.wav")
    dist    = loss_fn.forward(wav_ref, wav_out)
    print(dist)
"""

from __future__ import annotations

import inspect
from pathlib import Path

import librosa
import numpy as np
import torch

from audio_xai.models.dpam.models import DPAMNet

SAMPLE_RATE: int = 22050
_WEIGHTS_FILENAME = "dpam_pretrained.pth"


class DPAM:
    """Deep Perceptual Audio Metric (DPAM).

    Measures perceptual distance between two audio signals using a learned
    feature-loss network trained on just-noticeable-difference (JND) data.

    The model expects raw waveforms pre-processed with :func:`load_audio`.

    Args:
        model_path: Path to a ``.pth`` checkpoint produced by
            :meth:`save_weights` or converted from the original TF checkpoint.
            When ``None`` the class looks for ``dpam_pretrained.pth`` bundled
            alongside this module; if not found the model starts with random
            weights (useful for integration tests / fine-tuning).
        device: PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
            Defaults to CUDA when available, otherwise CPU.
        n_layers: Number of convolutional blocks in the loss network (14).
        base_channels: Base channel count for the first block (32).
        blk_channels: Channels double every this many layers (5).
        norm_type: Layer normalisation type – ``"SBN"``, ``"NM"``, ``"none"``.
        kernel_size: Conv kernel width (3).
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        n_layers: int = 14,
        base_channels: int = 32,
        blk_channels: int = 5,
        norm_type: str = "SBN",
        kernel_size: int = 3,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = DPAMNet(
            n_layers=n_layers,
            base_channels=base_channels,
            blk_channels=blk_channels,
            norm_type=norm_type,
            kernel_size=kernel_size,
        )

        resolved_path = self._resolve_weights(model_path)
        if resolved_path is not None:
            checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("state", checkpoint)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        wav_ref: np.ndarray,
        wav_deg: np.ndarray,
    ) -> float:
        """Return the DPAM perceptual distance between two audio arrays.

        Args:
            wav_ref: Reference waveform, shape ``[1, N]``, float32,
                sampled at 22 050 Hz (see :func:`load_audio`).
            wav_deg: Degraded waveform, same shape/format as ``wav_ref``.

        Returns:
            Non-negative scalar distance (lower = more similar).
        """
        with torch.no_grad():
            ref = torch.from_numpy(wav_ref).float().unsqueeze(0).to(self.device)
            deg = torch.from_numpy(wav_deg).float().unsqueeze(0).to(self.device)
            dist = self.model(ref, deg)
        return float(dist.item())

    def save_weights(self, path: str | Path) -> None:
        """Save model weights so they can be reloaded later."""
        torch.save({"state": self.model.state_dict()}, path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_weights(self, model_path: str | None) -> Path | None:
        if model_path is not None:
            p = Path(model_path)
            if not p.exists():
                raise FileNotFoundError(f"DPAM weights not found: {p}")
            return p
        # Look for bundled weights next to this source file
        bundled = Path(inspect.getfile(self.__class__)).parent / "weights" / _WEIGHTS_FILENAME
        if bundled.exists():
            return bundled
        return None


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def load_audio(path: str | Path) -> np.ndarray:
    """Load and pre-process a WAV file for use with :class:`DPAM`.

    Resamples to 22 050 Hz (mono), converts to the 16-bit float scale
    used by the original implementation, and returns an array of shape
    ``[1, N]`` with dtype ``float32``.

    Args:
        path: Path to any audio file supported by *librosa*.

    Returns:
        Float32 array of shape ``[1, N]``.
    """
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    # Match the 16-bit floating-point convention from the original code
    audio = np.round(audio.astype(np.float64) * 32768.0).astype(np.float32)
    return audio.reshape(1, -1)
