"""PyTorch Dataset for Google Speech Commands WAV samples.

Expected directory layout (produced by fetch_speech.py)::

    speech_commands_samples/
        yes_e9c9ef6a_nohohit_0.wav
        no_3f2af7cb_ab_2.wav
        ...

Filenames must follow the pattern ``{label}_{speaker_id}_{utterance_id}.wav``.
Labels are derived from the filename prefix; the dataset also builds an
integer encoding (alphabetical order) so DataLoader can collate batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


@dataclass
class SpeechCommandsConfig:
    root: Path
    clip_seconds: float = 1.0
    sample_rate: int = 16_000
    label_filter: str | None = None  # keep only this label, e.g. "yes"
    extensions: tuple[str, ...] = field(default_factory=lambda: (".wav",))
    max_samples: int | None = None


class SpeechCommandsDataset(Dataset):
    """Dataset wrapping pre-fetched Speech Commands WAV files.

    Returns ``(waveform [T], label_index)`` pairs.
    The label-to-index mapping is built from all labels found in the directory,
    sorted alphabetically, so it is deterministic across runs.
    """

    def __init__(self, cfg: SpeechCommandsConfig) -> None:
        self.cfg = cfg
        root = Path(cfg.root)
        if not root.is_dir():
            raise FileNotFoundError(f"Speech commands directory not found: {root}")

        paths: list[Path] = []
        for ext in cfg.extensions:
            paths.extend(sorted(root.glob(f"*{ext}")))

        if not paths:
            raise RuntimeError(f"No audio files found in {root}")

        # Build label vocabulary from filenames.
        labels_seen: set[str] = set()
        for p in paths:
            label = p.stem.split("_")[0]
            labels_seen.add(label)
        self.label2idx: dict[str, int] = {
            lbl: i for i, lbl in enumerate(sorted(labels_seen))
        }
        self.idx2label: list[str] = sorted(labels_seen)

        # Filter and cap.
        samples: list[tuple[Path, int]] = []
        for p in paths:
            label = p.stem.split("_")[0]
            if cfg.label_filter is not None and label != cfg.label_filter:
                continue
            samples.append((p, self.label2idx[label]))

        if cfg.max_samples is not None:
            samples = samples[: cfg.max_samples]

        if not samples:
            raise RuntimeError(
                f"No samples matched label_filter={cfg.label_filter!r} in {root}"
            )

        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def filename_at(self, index: int) -> str:
        """Return the source filename (stem + suffix) for the given dataset index."""
        return self._samples[index][0].name

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self._samples[index]
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