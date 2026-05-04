"""SONICS dataset loader for binary real/fake audio classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


@dataclass
class SonicsConfig:
    """Configuration for the SONICS binary audio dataset.

    Attributes:
        root: Root directory containing the ``real`` and ``fake`` subdirectories.
        clip_seconds: Target clip duration in seconds; longer clips are truncated and shorter ones are zero-padded.
        sample_rate: Output sample rate in Hz; audio is resampled when the file rate differs.
        real_subdir: Name of the subdirectory holding real audio files.
        fake_subdir: Name of the subdirectory holding fake/generated audio files.
        extensions: Tuple of file extensions to include when scanning for audio files.
    """

    root: Path
    clip_seconds: float = 10.0
    sample_rate: int = 16_000
    real_subdir: str = "real"
    fake_subdir: str = "fake"
    extensions: tuple[str, ...] = field(default_factory=lambda: (".wav", ".mp3", ".flac"))


class SonicsDataset(Dataset):
    """Binary dataset: label 0 = real, label 1 = fake.

    Expects the directory layout::

        root/
          real/  *.wav / *.mp3 / *.flac
          fake/  *.wav / *.mp3 / *.flac
    """

    def __init__(self, cfg: SonicsConfig) -> None:
        """Scan the dataset root and build an internal list of (path, label) pairs.

        Args:
            cfg: Dataset configuration specifying the root directory, sample rate,
                clip duration, subdirectory names, and supported file extensions.

        Raises:
            FileNotFoundError: If either the real or fake subdirectory does not exist.
            RuntimeError: If no audio files matching the configured extensions are found.
        """
        self.cfg = cfg
        self._samples: list[tuple[Path, int]] = []

        for label, subdir in ((0, cfg.real_subdir), (1, cfg.fake_subdir)):
            folder = cfg.root / subdir
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected directory: {folder}")
            for ext in cfg.extensions:
                for path in sorted(folder.glob(f"*{ext}")):
                    self._samples.append((path, label))

        if not self._samples:
            raise RuntimeError(f"No audio files found under {cfg.root}")

    def __len__(self) -> int:
        """Return the total number of audio samples in the dataset."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load, preprocess, and return the audio sample at the given index.

        Multi-channel audio is converted to mono by averaging channels. The
        waveform is resampled to ``cfg.sample_rate`` when necessary and then
        padded with zeros or truncated to exactly ``cfg.clip_seconds`` in length.

        Args:
            idx: Integer index into the dataset.

        Returns:
            A tuple ``(waveform, label)`` where ``waveform`` is a 1-D float
            tensor of shape ``[T]`` and ``label`` is 0 for real or 1 for fake.
        """
        path, label = self._samples[idx]
        waveform, sr = torchaudio.load(str(path))

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.cfg.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.cfg.sample_rate)

        target_len = int(self.cfg.clip_seconds * self.cfg.sample_rate)
        waveform = waveform.squeeze(0)
        if waveform.shape[0] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[0]))
        else:
            waveform = waveform[:target_len]

        return waveform, label
