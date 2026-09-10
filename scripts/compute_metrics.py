"""Compute CDPAM and kernel distance (KD) for attack run directories.

Per sample (written into the existing sample_*.json):
    cdpam     — Contrastive Deep Perceptual Audio Metric (lower = more similar)
    fad_dist  — L2 distance between log-mel embeddings
    kd        — polynomial kernel distance between log-mel embeddings

Skips a metric for a sample if the key already exists in the JSON
(use --force to recompute everything).

Usage:
    python scripts/compute_metrics.py [--device cuda] [--force] [RUN_DIR ...]

Default: all 9 attack run directories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_xai.metrics.fad import kernel_distance, polynomial_kernel

RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"

CDPAM_SR   = 22050
CDPAM_SECS = 5.0   # clip to 5 s to avoid OOM on long Sonics audio

ALL_RUNS: list[str] = [
    "attack/ast_epoch=4-step=9740_test_100_samples",
    "attack/vggish_epoch-epoch=009_test_100_samples",
    "attack/sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples",
    "pgd_attack/pgd_ast_epoch=4-step=9740_test_100_samples",
    "pgd_attack/pgd_vggish_epoch-epoch=009_test_100_samples",
    "pgd_attack/pgd_sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples",
    "xshift_attack/xshift_ast_epoch=4-step=9740_test_100_samples_shift",
    "xshift_attack/xshift_vggish_epoch-epoch=009_test_100_samples_shift",
    "xshift_attack/xshift_sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples_shift",
]


# ── Audio loading ─────────────────────────────────────────────────────────────

def _load_cdpam(path: Path) -> np.ndarray:
    """Load up to CDPAM_SECS seconds at CDPAM_SR, return [1, N] float32."""
    y, _ = librosa.load(str(path), sr=CDPAM_SR, mono=True, duration=CDPAM_SECS)
    return y[np.newaxis, :].astype(np.float32)


def _load_mono(path: Path, target_sr: int) -> torch.Tensor:
    """Load audio, resample to target_sr, return 1-D tensor."""
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.mean(dim=0)


# ── KD helpers ────────────────────────────────────────────────────────────────

def _mel_embedding(wav: torch.Tensor, sr: int, n_mels: int = 128) -> np.ndarray:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=512, n_mels=n_mels,
    )(wav.unsqueeze(0))
    return torch.log1p(mel).mean(dim=-1).squeeze(0).numpy()


def _kd_pair(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x[np.newaxis], y[np.newaxis]
    kxx = float(polynomial_kernel(x, x).item())
    kyy = float(polynomial_kernel(y, y).item())
    kxy = float(polynomial_kernel(x, y).item())
    return kxx + kyy - 2 * kxy


# ── Per-run processing ────────────────────────────────────────────────────────

def process_run(
    run_dir: Path,
    cdpam_fn,
    device: str,
    force: bool,
) -> None:
    results_path = run_dir / "results.json"
    audio_dir    = run_dir / "audio"
    if not results_path.exists() or not audio_dir.exists():
        print(f"  [skip] missing results.json or audio/ in {run_dir.name}")
        return

    sr = json.loads(results_path.read_text()).get("run", {}).get("sample_rate", 16_000)
    print(f"  sample_rate={sr} Hz")

    # stem → sample JSON path
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
            tqdm.write(f"  [warn] missing audio for {stem}")
            errors += 1
            continue

        jf = stem_to_json.get(stem)
        if jf is None:
            tqdm.write(f"  [warn] no sample JSON for {stem}")
            errors += 1
            continue

        sample = json.loads(jf.read_text())
        need_cdpam = force or "cdpam" not in sample
        need_kd    = force or "kd" not in sample

        if not need_cdpam and not need_kd:
            continue  # nothing to do for this sample

        try:
            if need_cdpam:
                ref = _load_cdpam(orig_path)
                deg = _load_cdpam(adv_path)
                min_len = min(ref.shape[1], deg.shape[1])
                ref, deg = ref[:, :min_len], deg[:, :min_len]
                with torch.no_grad():
                    score = cdpam_fn.forward(ref, deg)
                sample["cdpam"] = round(float(score.item()), 6)

            if need_kd:
                orig_wav = _load_mono(orig_path, sr)
                adv_wav  = _load_mono(adv_path,  sr)
                orig_emb = _mel_embedding(orig_wav, sr)
                adv_emb  = _mel_embedding(adv_wav,  sr)
                sample["fad_dist"] = round(float(np.linalg.norm(orig_emb - adv_emb)), 6)
                sample["kd"]       = round(_kd_pair(orig_emb, adv_emb), 6)

        except Exception as e:
            tqdm.write(f"  [error] {stem}: {e}")
            errors += 1
            continue

        jf.write_text(json.dumps(sample, indent=2))

    n_done = len(stems) - errors
    print(f"  Done: {n_done}/{len(stems)}  ({errors} errors)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force",  action="store_true",
                   help="Recompute even if the metric already exists in the JSON.")
    p.add_argument("runs", nargs="*", metavar="RUN_DIR",
                   help="Run dirs (absolute or relative to runs/). Default: all 9.")
    args = p.parse_args()

    dirs = [
        Path(r) if Path(r).is_absolute() else RUNS_ROOT / r
        for r in (args.runs or ALL_RUNS)
    ]

    print(f"Device : {args.device}")
    print(f"Runs   : {len(dirs)}")
    print("Loading CDPAM model…")
    import cdpam
    cdpam_fn = cdpam.CDPAM(dev=args.device)

    for run_dir in dirs:
        if not run_dir.is_dir():
            print(f"[skip] not found: {run_dir}")
            continue
        print(f"\n{'='*60}\nRun: {run_dir.name}")
        process_run(run_dir, cdpam_fn, args.device, args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
