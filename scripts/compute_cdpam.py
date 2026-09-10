"""Compute CDPAM (Contrastive Deep Perceptual Audio Metric) for attack runs.

For each sample in the specified run directories, loads original.wav and
adversarial.wav, computes CDPAM(original, adversarial), and writes the
result into the existing sample_*.json file as "cdpam".

Usage:
    python scripts/compute_cdpam.py [--device cuda] [RUN_DIR ...]

Default runs (3 attack runs):
    attack/ast_epoch=4-step=9740_test_100_samples
    attack/vggish_epoch-epoch=009_test_100_samples
    attack/sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples
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
CLIP_SECS  = 5.0          # CDPAM is memory-intensive; clip all audio to this length

DEFAULT_RUNS = [
    "attack/ast_epoch=4-step=9740_test_100_samples",
    "attack/vggish_epoch-epoch=009_test_100_samples",
    "attack/sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples",
]


def load_wav(path: Path, sr: int = CDPAM_SR, max_secs: float = CLIP_SECS) -> np.ndarray:
    """Load audio resampled to sr, clipped to max_secs, return [1, N] float32."""
    y, _ = librosa.load(str(path), sr=sr, mono=True, duration=max_secs)
    return y[np.newaxis, :].astype(np.float32)


def process_run(run_dir: Path, loss_fn, device: str) -> None:
    audio_root = run_dir / "audio"
    if not audio_root.exists():
        print(f"  [skip] no audio/ in {run_dir.name}")
        return

    # stem → json path
    stem_to_json: dict[str, Path] = {}
    for jf in run_dir.glob("sample_*.json"):
        d = json.loads(jf.read_text())
        stem = d.get("stem", "")
        if stem:
            stem_to_json[stem] = jf

    stems = sorted(p.name for p in audio_root.iterdir() if p.is_dir())
    errors = 0

    for stem in tqdm(stems, desc=run_dir.name, unit="stem"):
        orig_path = audio_root / stem / "original.wav"
        adv_path  = audio_root / stem / "adversarial.wav"

        if not orig_path.exists() or not adv_path.exists():
            tqdm.write(f"  [warn] missing audio for {stem}")
            errors += 1
            continue

        try:
            ref = load_wav(orig_path)
            deg = load_wav(adv_path)

            # CDPAM requires equal length
            min_len = min(ref.shape[1], deg.shape[1])
            ref = ref[:, :min_len]
            deg = deg[:, :min_len]

            with torch.no_grad():
                score = loss_fn.forward(ref, deg)
            cdpam_val = round(float(score.item()), 6)

        except Exception as e:
            tqdm.write(f"  [warn] CDPAM failed for {stem}: {e}")
            errors += 1
            continue

        jf = stem_to_json.get(stem)
        if jf is not None:
            d = json.loads(jf.read_text())
            d["cdpam"] = cdpam_val
            jf.write_text(json.dumps(d, indent=2))

    print(f"  Done: {len(stems)-errors}/{len(stems)}  ({errors} errors)")


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("runs", nargs="*", metavar="RUN_DIR",
                   help="Run dirs (absolute or relative to runs/). Default: 3 attack runs.")
    args = p.parse_args()

    dirs = [Path(r) if Path(r).is_absolute() else RUNS_ROOT / r
            for r in (args.runs or DEFAULT_RUNS)]

    print(f"Device : {args.device}")
    print(f"Runs   : {len(dirs)}")
    print("Loading CDPAM model…")

    import cdpam
    loss_fn = cdpam.CDPAM(dev=args.device)

    for run_dir in dirs:
        if not run_dir.exists():
            print(f"[skip] not found: {run_dir}")
            continue
        print(f"\n{'='*60}\nRun: {run_dir.name}")
        process_run(run_dir, loss_fn, args.device)

    print("\nDone.")


if __name__ == "__main__":
    main()
