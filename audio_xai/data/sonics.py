"""SONICS dataset loader for binary real/fake audio classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


@dataclass
class SonicsConfig:
    root: Path
    clip_seconds: float = 10.0
    sample_rate: int = 16_000
    real_subdir: str = "real"
    fake_subdir: str = "fake"
    extensions: tuple[str, ...] = field(default_factory=lambda: (".wav", ".mp3", ".flac"))
    max_per_class: int | None = None  # cap each class to this many samples (balanced dataset)
    csv_file: Path | None = None      # if set, load samples from this CSV instead of subdirs


class SonicsDataset(Dataset):
    """Binary dataset: label 0 = real, label 1 = fake.

    Can be loaded from a CSV with columns ``filepath`` (relative to cfg.root)
    and ``target`` (0/1), or from the directory layout::

        root/
          real/  *.wav / *.mp3 / *.flac
          fake/  *.wav / *.mp3 / *.flac
    """

    def __init__(self, cfg: SonicsConfig) -> None:
        self.cfg = cfg
        self._samples: list[tuple[Path, int]] = []

        if cfg.csv_file is not None:
            df = pd.read_csv(cfg.csv_file)
            for _, row in df.iterrows():
                self._samples.append((cfg.root / row["filepath"], int(row["target"])))
        else:
            for label, subdir in ((0, cfg.real_subdir), (1, cfg.fake_subdir)):
                folder = cfg.root / subdir
                if not folder.is_dir():
                    raise FileNotFoundError(f"Expected directory: {folder}")
                paths: list[Path] = []
                for ext in cfg.extensions:
                    paths.extend(sorted(folder.glob(f"*{ext}")))
                if cfg.max_per_class is not None:
                    paths = paths[: cfg.max_per_class]
                for path in paths:
                    self._samples.append((path, label))

        if not self._samples:
            raise RuntimeError(f"No audio files found under {cfg.root}")

    def __len__(self) -> int:
        return len(self._samples)

    def filename_at(self, index: int) -> str:
        """Return the source filename (stem + suffix) for the given dataset index."""
        return self._samples[index][0].name

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self._samples[index]
        target_len = int(self.cfg.clip_seconds * self.cfg.sample_rate)
        try:
            waveform, sr = torchaudio.load(str(path))
        except Exception as e:
            print(f"[SonicsDataset] skip unreadable file {path.name}: {e}", flush=True)
            return torch.zeros(target_len), label

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.cfg.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.cfg.sample_rate)

        waveform = waveform.squeeze(0)
        if waveform.shape[0] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[0]))
        else:
            waveform = waveform[:target_len]

        return waveform, label
