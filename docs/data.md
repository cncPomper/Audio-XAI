# Data

## Dataset layout

The project uses the **Sonics** dataset of real and AI-generated music. Two layouts are supported:

### Directory layout

```
data/
└── external/
    ├── real/          # real recordings (.wav / .mp3 / .flac)
    └── fake/          # AI-generated audio
```

Pass `--data-root /path/to/data/external` and optionally `--real-subdir real --fake-subdir fake`.

### CSV layout (preferred)

CSV files with two columns: `filepath` (relative to `--data-root`) and `target` (0 = real, 1 = fake).

```
data/
└── external/
    ├── train.csv
    ├── valid.csv
    ├── test.csv
    └── audio/          # all audio files referenced by the CSVs
```

Example CSV:

```csv
filepath,target
audio/real_001.wav,0
audio/fake_001.wav,1
audio/real_002.wav,0
```

Pass `--train-csv data/external/train.csv --val-csv data/external/valid.csv` to training scripts.

The attack and predict scripts use the split name (`--split test`) to locate `test.csv` inside `--data-root`.

## SonicsConfig / SonicsDataset

```python
from audio_xai.data.sonics import SonicsConfig, SonicsDataset
from pathlib import Path

cfg = SonicsConfig(
    root=Path("data/external"),
    clip_seconds=5.0,       # all clips trimmed / padded to this length
    sample_rate=16_000,
    real_subdir="real",     # ignored when csv_file is set
    fake_subdir="fake",
    max_per_class=None,     # cap per-class sample count (None = unlimited)
    csv_file=Path("data/external/train.csv"),  # takes priority over subdirs
)

dataset = SonicsDataset(cfg)
waveform, label = dataset[0]   # waveform: [T] float32, label: 0 or 1
```

`SonicsDataset.__getitem__` loads audio via `torchaudio`, resamples if needed, mono-mixes multi-channel files, and pads or trims to `clip_seconds`.

## Splits

The predict and attack scripts resolve splits as follows:

| `--split` | File looked up |
|-----------|---------------|
| `train`   | `{data_root}/train.csv` |
| `valid`   | `{data_root}/valid.csv` |
| `test`    | `{data_root}/test.csv`  |

If no CSV is found for a split, the script falls back to directory-mode scanning under `real_subdir` / `fake_subdir`.

## Balanced sampling

Both `predict.py` and `attack.py` draw balanced batches (equal real and fake samples) from whatever split is selected. Pass `--n-attack-samples 100` (for attack) or `--n-samples 500` (for predict) to cap the total. The selection is deterministic given `--seed`.