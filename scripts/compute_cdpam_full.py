"""Compute CDPAM on full audio for the 3 perceptual attack runs.

Audio files can be 5–300 s, so we process in 5-second chunks and average
the per-chunk CDPAM scores. This avoids GPU OOM while covering the whole file.

Overwrites any existing "cdpam" value in each sample_*.json.

Usage:
    python scripts/compute_cdpam_full.py [--device cuda] [RUN_DIR ...]

Default: the 3 attack/ run directories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUNS_ROOT  = Path(__file__).resolve().parent.parent / "runs"
CDPAM_SR   = 22050
CHUNK_SECS = 5.0

DEFAULT_RUNS = [
    "attack/ast_epoch=4-step=9740_test_100_samples",
    "attack/vggish_epoch-epoch=009_test_100_samples",
    "attack/sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples",
]


def load_full(path: Path) -> np.ndarray:
    """Load full audio resampled to CDPAM_SR, return [N] float32."""
    y, _ = librosa.load(str(path), sr=CDPAM_SR, mono=True)
    return y.astype(np.float32)


def cdpam_chunked(cdpam_fn, ref: np.ndarray, deg: np.ndarray) -> float:
    """Average CDPAM over non-overlapping CHUNK_SECS windows."""
    chunk_len = int(CHUNK_SECS * CDPAM_SR)
    # Align both signals to the same length
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]

    scores = []
    for start in range(0, n, chunk_len):
        r = ref[start: start + chunk_len]
        d = deg[start: start + chunk_len]
        if len(r) < 1000:   # skip very short trailing chunk
            continue
        # Pad to chunk_len if the last chunk is shorter
        if len(r) < chunk_len:
            r = np.pad(r, (0, chunk_len - len(r)))
            d = np.pad(d, (0, chunk_len - len(d)))
        r_t = torch.from_numpy(r[np.newaxis]).float()
        d_t = torch.from_numpy(d[np.newaxis]).float()
        with torch.no_grad():
            s = cdpam_fn.forward(r_t.numpy(), d_t.numpy())
        scores.append(float(s.item()))

    return float(np.mean(scores)) if scores else 0.0


def process_run(run_dir: Path, cdpam_fn) -> None:
    audio_dir = run_dir / "audio"
    if not audio_dir.exists():
        print(f"  [skip] no audio/ in {run_dir.name}")
        return

    # stem → JSON path
    stem_to_json: dict[str, Path] = {}
    for jf in run_dir.glob("sample_*.json"):
        parts = jf.stem.split("_", 2)
        if len(parts) == 3:
            stem_to_json[parts[2]] = jf

    stems = sorted(p.name for p in audio_dir.iterdir() if p.is_dir())
    errors = 0

    for stem in tqdm(stems, desc=run_dir.name, unit="stem"):
        orig_path = audio_dir / stem / "original.wav"
        adv_path  = audio_dir / stem / "adversarial.wav"
        if not orig_path.exists() or not adv_path.exists():
            tqdm.write(f"  [warn] missing audio: {stem}")
            errors += 1
            continue

        jf = stem_to_json.get(stem)
        if jf is None:
            tqdm.write(f"  [warn] no sample JSON: {stem}")
            errors += 1
            continue

        try:
            ref = load_full(orig_path)
            deg = load_full(adv_path)
            val = cdpam_chunked(cdpam_fn, ref, deg)
        except Exception as e:
            tqdm.write(f"  [error] {stem}: {e}")
            errors += 1
            continue

        sample = json.loads(jf.read_text())
        sample["cdpam"] = round(val, 6)
        jf.write_text(json.dumps(sample, indent=2))

    print(f"  Done: {len(stems)-errors}/{len(stems)}  ({errors} errors)")


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("runs", nargs="*", metavar="RUN_DIR",
                   help="Run dirs (absolute or relative to runs/). Default: 3 attack/ runs.")
    args = p.parse_args()

    dirs = [
        Path(r) if Path(r).is_absolute() else RUNS_ROOT / r
        for r in (args.runs or DEFAULT_RUNS)
    ]

    print(f"Device     : {args.device}")
    print(f"Chunk size : {CHUNK_SECS}s at {CDPAM_SR} Hz")
    print(f"Runs       : {len(dirs)}")
    print("Loading CDPAM model…")
    import cdpam
    cdpam_fn = cdpam.CDPAM(dev=args.device)

    for run_dir in dirs:
        if not run_dir.is_dir():
            print(f"[skip] not found: {run_dir}")
            continue
        print(f"\n{'='*60}\nRun: {run_dir.name}")
        process_run(run_dir, cdpam_fn)

    print("\nDone.")


if __name__ == "__main__":
    main()
