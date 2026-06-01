"""
analyze_audio_easy_hard.py

Standalone script #2.

Cel:
- bierze pliki easy_sample_names.txt i hard_sample_names.txt wygenerowane przez script #1,
- wyszukuje odpowiadające pliki audio w folderze użytkownika,
- liczy cechy audio,
- porównuje easy vs hard,
- zapisuje CSV i wykresy, żeby zobaczyć prawidłowości:
    co odróżnia próbki łatwe od trudnych do atakowania atrybucyjnego.

Wymagania:
    pip install numpy pandas matplotlib scipy soundfile

Przykład:
    python analyze_audio_easy_hard.py \
        --audio-root /path/to/audio_dataset \
        --easy-names case_selection/easy_sample_names.txt \
        --hard-names case_selection/hard_sample_names.txt \
        --output-dir audio_case_analysis

Dla struktury:
    audio_root/sample_name/original.wav

użyj:
    python analyze_audio_easy_hard.py \
        --audio-root /path/to/audio_dataset \
        --easy-names case_selection/easy_sample_names.txt \
        --hard-names case_selection/hard_sample_names.txt \
        --output-dir audio_case_analysis \
        --layout sample_dirs \
        --audio-filename original.wav

Matching:
- domyślnie skrypt dopasowuje po file.stem == nazwa próbki,
- jeśli rozszerzenia lub nazwy są dłuższe, użyj --match-mode contains.

Uwaga:
- Najlepiej używać WAV/FLAC/OGG obsługiwanych przez soundfile.
- MP3 zależy od lokalnej instalacji libsndfile; jeśli nie działa, przekonwertuj do WAV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import soundfile as sf
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: soundfile. Install with: pip install soundfile"
    ) from exc

try:
    from scipy import signal, stats
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: scipy. Install with: pip install scipy"
    ) from exc


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc", ".mp3", ".m4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze audio features for easy vs hard attribution-attack samples."
    )
    parser.add_argument(
        "--audio-root", required=True, type=Path, help="Root folder with audio files."
    )
    parser.add_argument(
        "--easy-names",
        required=True,
        type=Path,
        help="TXT file with easy sample names.",
    )
    parser.add_argument(
        "--hard-names",
        required=True,
        type=Path,
        help="TXT file with hard sample names.",
    )
    parser.add_argument("--output-dir", default=Path("audio_case_analysis"), type=Path)
    parser.add_argument(
        "--match-mode",
        choices=["exact", "contains"],
        default="exact",
        help="exact: file.stem == sample name; contains: sample name is substring of file.stem.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search audio-root. Recommended for nested datasets.",
    )
    parser.add_argument(
        "--frame-ms",
        default=25.0,
        type=float,
        help="Frame length in ms for frame-level features.",
    )
    parser.add_argument(
        "--hop-ms",
        default=10.0,
        type=float,
        help="Hop length in ms for frame-level features.",
    )
    parser.add_argument(
        "--silence-db",
        default=-40.0,
        type=float,
        help="Frame RMS below this dBFS is treated as silence.",
    )
    parser.add_argument(
        "--max-seconds",
        default=None,
        type=float,
        help="Optionally crop audio to first N seconds for faster analysis.",
    )
    parser.add_argument(
        "--layout",
        choices=["flat", "sample_dirs"],
        default="flat",
        help=(
            "flat: search audio files by file stem; "
            "sample_dirs: expect audio_root/sample_name/audio_filename."
        ),
    )
    parser.add_argument(
        "--audio-filename",
        default="original.wav",
        help="Audio filename inside each sample directory when --layout sample_dirs is used.",
    )
    return parser.parse_args()


def read_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return list(dict.fromkeys(names))


def index_audio_files(audio_root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in audio_root.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return files


def find_audio_for_name(
    name: str, files: list[Path], match_mode: str
) -> Optional[Path]:
    name = str(name)

    # Prefer exact stem match.
    exact = [p for p in files if p.stem == name]
    if exact:
        return sorted(exact, key=lambda p: len(str(p)))[0]

    if match_mode == "contains":
        contains = [p for p in files if name in p.stem]
        if contains:
            return sorted(contains, key=lambda p: len(str(p)))[0]

    return None


def load_audio_mono(
    path: Path, max_seconds: Optional[float] = None
) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True)

    if audio.size == 0:
        raise ValueError("Empty audio file")

    y = audio.mean(axis=1).astype(np.float64)

    if max_seconds is not None:
        n = int(max_seconds * sr)
        y = y[:n]

    # Avoid NaN/inf.
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # If integer-scaled data somehow exceeds [-1, 1], normalize safely.
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 1.5:
        y = y / peak

    return y, sr


def frame_signal(y: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(y) < frame_len:
        pad = frame_len - len(y)
        y = np.pad(y, (0, pad), mode="constant")

    n_frames = 1 + (len(y) - frame_len) // hop_len
    if n_frames <= 0:
        return y[:frame_len][None, :]

    shape = (n_frames, frame_len)
    strides = (y.strides[0] * hop_len, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    return frames.copy()


def safe_db(x: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(x, eps))


def spectral_features(y: np.ndarray, sr: int) -> dict[str, float]:
    if len(y) < 8:
        return {}

    nperseg = min(2048, max(256, 2 ** int(np.floor(np.log2(len(y))))))
    noverlap = nperseg // 2

    freqs, times, Zxx = signal.stft(
        y,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )

    mag = np.abs(Zxx) + 1e-12
    power = mag**2
    power_sum = power.sum(axis=0) + 1e-12

    centroid = (freqs[:, None] * power).sum(axis=0) / power_sum
    bandwidth = np.sqrt(
        (((freqs[:, None] - centroid[None, :]) ** 2) * power).sum(axis=0) / power_sum
    )

    cumulative = np.cumsum(power, axis=0)
    rolloff_threshold = 0.85 * power_sum
    rolloff_idx = np.array(
        [
            np.searchsorted(cumulative[:, i], rolloff_threshold[i])
            for i in range(power.shape[1])
        ]
    )
    rolloff_idx = np.clip(rolloff_idx, 0, len(freqs) - 1)
    rolloff = freqs[rolloff_idx]

    flatness = np.exp(np.mean(np.log(mag), axis=0)) / (np.mean(mag, axis=0) + 1e-12)

    # Band energy ratios. These are coarse but interpretable for audio/speech/music.
    total_energy = power.sum() + 1e-12
    low = power[freqs < 500].sum() / total_energy
    mid = power[(freqs >= 500) & (freqs < 4000)].sum() / total_energy
    high = power[freqs >= 4000].sum() / total_energy

    # Spectral flux: frame-to-frame magnitude change.
    if mag.shape[1] > 1:
        mag_norm = mag / (np.linalg.norm(mag, axis=0, keepdims=True) + 1e-12)
        flux = np.sqrt(np.sum(np.diff(mag_norm, axis=1) ** 2, axis=0))
        spectral_flux_mean = float(np.mean(flux))
        spectral_flux_std = float(np.std(flux))
    else:
        spectral_flux_mean = 0.0
        spectral_flux_std = 0.0

    return {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "spectral_rolloff85_mean": float(np.mean(rolloff)),
        "spectral_rolloff85_std": float(np.std(rolloff)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "spectral_flatness_std": float(np.std(flatness)),
        "low_band_energy_ratio": float(low),
        "mid_band_energy_ratio": float(mid),
        "high_band_energy_ratio": float(high),
        "spectral_flux_mean": spectral_flux_mean,
        "spectral_flux_std": spectral_flux_std,
    }


def extract_audio_features(
    path: Path,
    frame_ms: float,
    hop_ms: float,
    silence_db: float,
    max_seconds: Optional[float],
) -> dict[str, float | str]:
    y, sr = load_audio_mono(path, max_seconds=max_seconds)

    duration = len(y) / sr if sr else 0.0
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    rms = float(np.sqrt(np.mean(y**2))) if len(y) else 0.0
    rms_db = float(safe_db(rms))
    peak_db = float(safe_db(peak))
    crest_factor = float(peak / (rms + 1e-12))

    frame_len = max(8, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    frames = frame_signal(y, frame_len, hop_len)

    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    frame_rms_db = safe_db(frame_rms)
    silence_ratio = float(np.mean(frame_rms_db < silence_db))

    # Dynamic range estimated from frame RMS percentiles.
    p95 = float(np.percentile(frame_rms_db, 95))
    p05 = float(np.percentile(frame_rms_db, 5))
    dynamic_range_db = p95 - p05

    # Zero-crossing rate.
    signs = np.signbit(frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)

    feats = {
        "path": str(path),
        "sample_rate": sr,
        "duration_sec": float(duration),
        "rms": rms,
        "rms_db": rms_db,
        "peak": peak,
        "peak_db": peak_db,
        "crest_factor": crest_factor,
        "frame_rms_db_mean": float(np.mean(frame_rms_db)),
        "frame_rms_db_std": float(np.std(frame_rms_db)),
        "dynamic_range_db": float(dynamic_range_db),
        "silence_ratio": silence_ratio,
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
    }

    feats.update(spectral_features(y, sr))
    return feats


def cohen_d(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    y = pd.to_numeric(y, errors="coerce").dropna().astype(float)

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx, ny = len(x), len(y)
    pooled = np.sqrt(
        ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    )
    if pooled == 0 or np.isnan(pooled):
        return np.nan

    return float((x.mean() - y.mean()) / pooled)


def compare_groups(features: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in {"sample_rate"}]

    rows = []
    easy = features[features["group"] == "easy"]
    hard = features[features["group"] == "hard"]

    for col in numeric_cols:
        x = pd.to_numeric(easy[col], errors="coerce").dropna()
        y = pd.to_numeric(hard[col], errors="coerce").dropna()

        if len(x) == 0 or len(y) == 0:
            continue

        try:
            stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
        except Exception:
            p_value = np.nan

        rows.append(
            {
                "feature": col,
                "easy_mean": x.mean(),
                "hard_mean": y.mean(),
                "easy_median": x.median(),
                "hard_median": y.median(),
                "mean_diff_easy_minus_hard": x.mean() - y.mean(),
                "median_diff_easy_minus_hard": x.median() - y.median(),
                "cohen_d_easy_minus_hard": cohen_d(x, y),
                "mannwhitney_p": p_value,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_cohen_d"] = out["cohen_d_easy_minus_hard"].abs()
        out = out.sort_values("abs_cohen_d", ascending=False)
    return out


def plot_effect_sizes(comparison: pd.DataFrame, out_dir: Path, top_n: int = 15) -> None:
    if comparison.empty:
        return

    top = comparison.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(top))))
    ax.tick_params(axis="y", labelsize=14)
    ax.barh(top["feature"], top["cohen_d_easy_minus_hard"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Cohen's d: easy minus hard", fontsize=16)
    # ax.set_title("Strongest differentiating audio features for easy vs hard samples")
    plt.tight_layout()
    plt.savefig(out_dir / "01_audio_feature_effect_sizes.png", dpi=300)
    plt.show()


def plot_boxplots(
    features: pd.DataFrame, comparison: pd.DataFrame, out_dir: Path, top_n: int = 8
) -> None:
    if comparison.empty:
        return

    selected = comparison.head(top_n)["feature"].tolist()

    for feature in selected:
        easy_vals = pd.to_numeric(
            features.loc[features["group"] == "easy", feature], errors="coerce"
        ).dropna()
        hard_vals = pd.to_numeric(
            features.loc[features["group"] == "hard", feature], errors="coerce"
        ).dropna()

        if len(easy_vals) == 0 or len(hard_vals) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot([easy_vals, hard_vals], labels=["easy", "hard"])
        ax.set_title(f"Easy vs hard: {feature}")
        ax.set_ylabel(feature)
        plt.tight_layout()
        safe_name = feature.replace("/", "_").replace(" ", "_")
        plt.savefig(out_dir / f"02_boxplot_{safe_name}.png", dpi=300)
        plt.show()


def write_markdown_summary(
    comparison: pd.DataFrame,
    features: pd.DataFrame,
    missing: pd.DataFrame,
    out_dir: Path,
) -> None:
    lines = []
    lines.append("# Audio analysis: easy vs hard samples\n")
    lines.append(
        "This report compares audio features for globally easy and hard attribution-attack samples.\n"
    )
    lines.append(
        "Hard samples are the ones previously selected as attribution-stable: prediction can be preserved, but the attribution map does not change easily.\n"
    )

    lines.append("## Counts\n")
    lines.append(
        f"- Easy audio files processed: {int((features['group'] == 'easy').sum()) if not features.empty else 0}"
    )
    lines.append(
        f"- Hard audio files processed: {int((features['group'] == 'hard').sum()) if not features.empty else 0}"
    )
    lines.append(f"- Missing audio files: {len(missing)}\n")

    if not comparison.empty:
        lines.append("## Top differentiating audio features\n")
        for _, row in comparison.head(10).iterrows():
            direction = (
                "higher for easy"
                if row["mean_diff_easy_minus_hard"] > 0
                else "higher for hard"
            )
            lines.append(
                f"- `{row['feature']}`: {direction}; "
                f"easy_mean={row['easy_mean']:.4g}, hard_mean={row['hard_mean']:.4g}, "
                f"Cohen_d={row['cohen_d_easy_minus_hard']:.3f}, "
                f"Mann-Whitney p={row['mannwhitney_p']:.3g}"
            )
        lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append(
        "- Large absolute Cohen's d suggests a feature that separates easy and hard samples strongly."
    )
    lines.append("- Positive Cohen's d means the feature is higher for easy samples.")
    lines.append("- Negative Cohen's d means the feature is higher for hard samples.")
    lines.append(
        "- With only top 10 vs top 10, p-values are exploratory, not definitive."
    )
    lines.append(
        "- Treat this as hypothesis generation: use it to identify patterns worth checking on a larger set.\n"
    )

    (out_dir / "audio_easy_hard_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    easy_names = read_names(args.easy_names)
    hard_names = read_names(args.hard_names)

    if args.layout == "sample_dirs":
        audio_files = []
        print(
            f"Using sample_dirs layout: "
            f"{args.audio_root} / <sample_name> / {args.audio_filename}"
        )
    else:
        audio_files = index_audio_files(args.audio_root, recursive=args.recursive)
        print(f"Indexed {len(audio_files)} audio files from {args.audio_root}")

    rows = []
    missing_rows = []

    for group, names in [("easy", easy_names), ("hard", hard_names)]:
        for name in names:
            if args.layout == "sample_dirs":
                candidate = args.audio_root / name / args.audio_filename
                path = candidate if candidate.exists() else None
            else:
                path = find_audio_for_name(name, audio_files, args.match_mode)

            if path is None:
                expected = (
                    str(args.audio_root / name / args.audio_filename)
                    if args.layout == "sample_dirs"
                    else "no matching audio file"
                )
                missing_rows.append(
                    {
                        "sample": name,
                        "group": group,
                        "reason": "not_found",
                        "expected": expected,
                    }
                )
                print(f"[MISSING] {group}: {name} -> {expected}")
                continue

            try:
                feats = extract_audio_features(
                    path=path,
                    frame_ms=args.frame_ms,
                    hop_ms=args.hop_ms,
                    silence_db=args.silence_db,
                    max_seconds=args.max_seconds,
                )
                feats["sample"] = name
                feats["group"] = group
                rows.append(feats)
                print(f"[OK] {group}: {name} -> {path}")
            except Exception as exc:
                missing_rows.append(
                    {
                        "sample": name,
                        "group": group,
                        "path": str(path),
                        "reason": str(exc),
                    }
                )
                print(f"[ERROR] {group}: {name} -> {path}: {exc}")

    features = pd.DataFrame(rows)
    missing = pd.DataFrame(missing_rows)

    features.to_csv(args.output_dir / "01_audio_features_easy_hard.csv", index=False)
    missing.to_csv(
        args.output_dir / "02_missing_or_failed_audio_files.csv", index=False
    )

    if features.empty:
        print(
            "No audio features extracted. Check matching mode, audio root, and file formats."
        )
        return

    comparison = compare_groups(features)
    comparison.to_csv(
        args.output_dir / "03_easy_vs_hard_audio_feature_comparison.csv", index=False
    )

    plot_effect_sizes(comparison, args.output_dir)
    plot_boxplots(features, comparison, args.output_dir)
    write_markdown_summary(comparison, features, missing, args.output_dir)

    print("\nSaved outputs to:", args.output_dir.resolve())
    print("\nTop differentiating features:")
    with pd.option_context(
        "display.max_rows", 20, "display.max_columns", None, "display.width", 200
    ):
        print(comparison.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
