"""Compute FAD-related metrics for an existing pgd_attack run.

Per sample (added to each sample_*.json):
    fad_dist  — L2 distance between log-mel embeddings of original vs adversarial
    kd        — polynomial kernel distance k(orig, orig) + k(adv, adv) − 2·k(orig, adv)

Dataset level (added to results.json):
    kernel_distance_mean / kernel_distance_std  — KID over all samples

Usage:
    python scripts/compute_fad.py \\
        runs/pgd_attack/pgd_vggish_epoch-epoch=009_test_100_samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from audio_xai.metrics.audio_metrics import AudioMetricsData
from audio_xai.metrics.fad import kernel_distance, polynomial_kernel


def mel_embedding(wav: torch.Tensor, sr: int, n_mels: int = 128) -> np.ndarray:
    """Log-mel mean embedding, shape (n_mels,)."""
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=512, n_mels=n_mels,
    )(wav.unsqueeze(0) if wav.ndim == 1 else wav)
    return torch.log1p(mel).mean(dim=-1).squeeze(0).numpy()


def _kernel_dist_pair(x: np.ndarray, y: np.ndarray) -> float:
    """Polynomial kernel distance between two single embeddings."""
    x = x[np.newaxis]   # (1, d)
    y = y[np.newaxis]
    kxx = float(polynomial_kernel(x, x).item())
    kyy = float(polynomial_kernel(y, y).item())
    kxy = float(polynomial_kernel(x, y).item())
    return kxx + kyy - 2 * kxy


def load_mono(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.mean(dim=0)   # mono


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--n-mels", type=int, default=128)
    args = p.parse_args()

    run_dir   = args.run_dir
    audio_dir = run_dir / "audio"
    json_path = run_dir / "results.json"

    if not audio_dir.exists():
        raise FileNotFoundError(f"audio/ not found in {run_dir}")
    if not json_path.exists():
        raise FileNotFoundError(f"results.json not found in {run_dir}")

    with open(json_path) as f:
        results = json.load(f)

    sr = results.get("run", {}).get("sample_rate", 16_000)
    print(f"Run dir    : {run_dir}")
    print(f"Sample rate: {sr} Hz  |  n_mels: {args.n_mels}")

    # Build index: stem → sample JSON path
    sample_jsons: dict[str, Path] = {}
    for jp in run_dir.glob("sample_*.json"):
        # filename like sample_0001_fake_50028_suno_1.json → stem = fake_50028_suno_1
        parts = jp.stem.split("_", 2)   # ["sample", "0001", "fake_50028_suno_1"]
        if len(parts) == 3:
            sample_jsons[parts[2]] = jp

    orig_amd = AudioMetricsData(store_embeddings=True)
    adv_amd  = AudioMetricsData(store_embeddings=True)

    n_ok = 0
    for sample_dir in sorted(audio_dir.iterdir()):
        orig_path = sample_dir / "original.wav"
        adv_path  = sample_dir / "adversarial.wav"
        if not orig_path.exists() or not adv_path.exists():
            print(f"  [skip] {sample_dir.name} — missing wav")
            continue
        try:
            orig_wav = load_mono(orig_path, sr)
            adv_wav  = load_mono(adv_path,  sr)
            orig_emb = mel_embedding(orig_wav, sr, args.n_mels)
            adv_emb  = mel_embedding(adv_wav,  sr, args.n_mels)

            # Per-sample metrics
            fad_dist = float(np.linalg.norm(orig_emb - adv_emb))
            kd_pair  = float(_kernel_dist_pair(orig_emb, adv_emb))

            # Patch matching sample JSON
            stem = sample_dir.name
            if stem in sample_jsons:
                jp = sample_jsons[stem]
                with open(jp) as f:
                    sd = json.load(f)
                sd["fad_dist"] = round(fad_dist, 6)
                sd["kd"]       = round(kd_pair,  6)
                with open(jp, "w") as f:
                    json.dump(sd, f, indent=2)
            else:
                print(f"  [warn] no sample JSON found for {stem}")

            # Accumulate for dataset-level KID
            orig_amd.add(orig_emb)
            adv_amd.add(adv_emb)
            n_ok += 1
            print(f"  {stem}: fad_dist={fad_dist:.4f}  kd={kd_pair:.6f}")

        except Exception as e:
            print(f"  [error] {sample_dir.name}: {e}")

    print(f"\nEmbeddings computed: {n_ok} samples")

    # Dataset-level KID
    if orig_amd.n >= 2:
        kd_res  = kernel_distance(orig_amd, adv_amd)
        kd_mean = kd_res["kernel_distance_mean"]
        kd_std  = kd_res["kernel_distance_std"]
        print(f"Dataset KID  mean={kd_mean:.6f}  std={kd_std:.6f}")
        results.setdefault("attack_summary", {})["kernel_distance_mean"] = kd_mean
        results["attack_summary"]["kernel_distance_std"]  = kd_std
        results.setdefault("perceptual_means", {})["kernel_distance_mean"] = kd_mean
        results["perceptual_means"]["kernel_distance_std"]  = kd_std
    else:
        print("Not enough samples for dataset KID.")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Updated → {json_path}")


if __name__ == "__main__":
    main()