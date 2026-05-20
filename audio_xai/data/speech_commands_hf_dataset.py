"""HuggingFace-backed PyTorch Dataset for Google Speech Commands.

Designed for training/fine-tuning classifiers on the full dataset
(~1 s clips, 16 kHz, 35 classes in v0.02).  Uses the non-streaming
HuggingFace Dataset with automatic caching — large but shuffleable.

Usage::

    from audio_xai.data.speech_commands_hf_dataset import (
        SpeechCommandsHFConfig, SpeechCommandsHFDataset, N_CLASSES_V002
    )

    train_ds = SpeechCommandsHFDataset(SpeechCommandsHFConfig(split="train"))
    val_ds   = SpeechCommandsHFDataset(SpeechCommandsHFConfig(split="validation"))
    test_ds  = SpeechCommandsHFDataset(SpeechCommandsHFConfig(split="test"))

    print(train_ds.label_names)   # ['backward', 'bed', 'bird', ...]
    wav, label = train_ds[0]      # wav: [T], label: int
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio
from torch.utils.data import Dataset

N_CLASSES_V001 = 30
N_CLASSES_V002 = 35


@dataclass
class SpeechCommandsHFConfig:
    version: str = "v0.02"          # "v0.01" | "v0.02"
    split: str = "train"            # "train" | "validation" | "test"
    sample_rate: int = 16_000
    clip_seconds: float = 1.0
    skip_unknown: bool = False      # filter out "_unknown_" label
    skip_silence: bool = False      # filter out "_silence_" label
    cache_dir: str | None = None    # HuggingFace cache directory


class SpeechCommandsHFDataset(Dataset):
    """Full Speech Commands dataset loaded from HuggingFace Hub.

    Each sample is a (waveform [T], label_index) pair.
    The label vocabulary is sorted alphabetically and exposed as
    ``self.label_names`` and ``self.label2idx``.
    """

    def __init__(self, cfg: SpeechCommandsHFConfig) -> None:
        self.cfg = cfg
        from datasets import Audio, load_dataset

        print(f"Loading google/speech_commands {cfg.version} [{cfg.split}] …")
        raw = load_dataset(
            "google/speech_commands",
            cfg.version,
            split=cfg.split,
            trust_remote_code=True,
            cache_dir=cfg.cache_dir,
        )
        # Cast audio column to target sample rate (HuggingFace handles resampling).
        raw = raw.cast_column("audio", Audio(sampling_rate=cfg.sample_rate))

        self.label_names: list[str] = raw.features["label"].names
        self.label2idx: dict[str, int] = {n: i for i, n in enumerate(self.label_names)}
        self.n_classes: int = len(self.label_names)

        # Optional filtering
        skip_indices: set[int] = set()
        if cfg.skip_unknown and "_unknown_" in self.label2idx:
            skip_indices.add(self.label2idx["_unknown_"])
        if cfg.skip_silence and "_silence_" in self.label2idx:
            skip_indices.add(self.label2idx["_silence_"])

        if skip_indices:
            raw = raw.filter(lambda ex: ex["label"] not in skip_indices)

        self._raw = raw
        self._target_len = int(cfg.clip_seconds * cfg.sample_rate)

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self._raw[index]
        array = sample["audio"]["array"]           # numpy float32 [T]
        sr    = sample["audio"]["sampling_rate"]   # should already be cfg.sample_rate
        label = int(sample["label"])

        wav = torch.from_numpy(array).float()

        # Resample if needed (safety guard — cast_column should handle this).
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)

        # Pad or crop to fixed length.
        T = self._target_len
        if wav.shape[0] < T:
            wav = torch.nn.functional.pad(wav, (0, T - wav.shape[0]))
        else:
            wav = wav[:T]

        return wav, label