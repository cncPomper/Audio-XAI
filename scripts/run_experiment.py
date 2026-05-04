#!/usr/bin/env python3
"""
Perceptual XAI Fragility Experiment
=====================================
Assesses how easily Grad-CAM explanations can be manipulated on AST and VGGish
classifiers while (a) keeping predictions intact and (b) keeping perturbations
below the psychoacoustic masking threshold.

For each sample the script:
  1. Records the original Grad-CAM heatmap
  2. Runs perceptual_xai_attack (Adam, second-order gradients)
  3. Measures explanation-similarity metrics and perturbation audibility

Outputs:
  {out_dir}/results.csv         — per-sample metrics table
  {out_dir}/summary.json        — aggregate statistics
  {out_dir}/convergence.png     — mean attack-loss curves ± 1 std
  {out_dir}/distributions.png   — histograms of the three similarity metrics

Usage:
  python scripts/run_experiment.py \\
      --data-root /path/to/sonics \\
      --model ast \\
      --checkpoint runs/ast/version_0/checkpoints/best.ckpt \\
      --n-samples 50 \\
      --out-dir reports/experiment_ast

  # Without a checkpoint (uses pretrained AudioSet weights for AST):
  python scripts/run_experiment.py \\
      --data-root /path/to/sonics \\
      --model ast \\
      --n-samples 20

SLURM example:
  sbatch --gres=gpu:1 --mem=32G --time=8:00:00 \\
      --wrap="python scripts/run_experiment.py \\
          --data-root /path/to/sonics --model ast --n-samples 200"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Make the package importable when run directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_xai.attacks.perceptual_xai_attack import (
    AttackConfig,
    heatmap_ssim,
    perceptual_xai_attack,
    topk_overlap,
)
from audio_xai.data.sonics import SonicsConfig, SonicsDataset
from audio_xai.models.ast_binary import ASTBinary
from audio_xai.models.lit_module import RealFakeLitModule, equal_error_rate
from audio_xai.models.vggish_binary import VGGishBinary


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(name: str, checkpoint: Path | None) -> torch.nn.Module:
    if name == "ast":
        model = ASTBinary(pretrained=True)
    elif name == "vggish":
        model = VGGishBinary()
    else:
        raise ValueError(f"Unknown model: {name!r}. Choose 'ast' or 'vggish'.")

    if checkpoint is not None:
        lit = RealFakeLitModule.load_from_checkpoint(
            str(checkpoint),
            model=model,
            strict=False,
            map_location="cpu",
        )
        model = lit.model
        print(f"[checkpoint] loaded {checkpoint}")

    return model


def _snr_db(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """Signal-to-noise ratio of the perturbation in dB."""
    sig_pow = signal.pow(2).mean().item()
    noi_pow = noise.pow(2).mean().item()
    if noi_pow < 1e-14:
        return float("inf")
    return 10.0 * math.log10(sig_pow / (noi_pow + 1e-14))


@torch.no_grad()
def _baseline_metrics(model: torch.nn.Module, loader: DataLoader, device: str) -> dict:
    """Accuracy, AUROC, and EER on the given DataLoader."""
    from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

    acc_m = BinaryAccuracy().to(device)
    auroc_m = BinaryAUROC().to(device)
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for wav, label in tqdm(loader, desc="Baseline eval", leave=False):
        wav, label = wav.to(device), label.to(device)
        logits = model(wav)
        probs = logits.softmax(-1)[:, 1]
        preds = logits.argmax(-1)
        acc_m.update(preds, label)
        auroc_m.update(probs, label)
        all_scores.append(probs.cpu())
        all_labels.append(label.cpu())

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    return {
        "accuracy": round(acc_m.compute().item(), 4),
        "auroc": round(auroc_m.compute().item(), 4),
        "eer": round(equal_error_rate(scores, labels), 4),
    }


def _save_convergence_plot(histories: list[list[dict]], model_name: str, n: int, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    keys = ["loss_explain", "loss_audibility", "loss_pred", "cos_sim"]
    titles = ["Explanation loss (↓)", "Audibility loss (↓)", "Prediction loss (↓)", "Cosine similarity (↓)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, key, title in zip(axes.flat, keys, titles):
        curves = [[h[key] for h in hist] for hist in histories if hist]
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        steps = [h["step"] for h in histories[0][:min_len]]
        mean, std = arr.mean(0), arr.std(0)
        ax.plot(steps, mean, lw=2, label="mean")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.25, label="±1 std")
        ax.set_title(title)
        ax.set_xlabel("Attack step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Attack convergence — {model_name}  (n={n})", fontsize=13)
    fig.tight_layout()
    path = out_dir / "convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot → {path}")


def _save_distribution_plot(df: pd.DataFrame, model_name: str, n: int, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    df["cosine_sim"].hist(bins=20, ax=axes[0], edgecolor="black", color="steelblue")
    axes[0].set_title("Cosine similarity (orig vs adv CAM)\nlower = more manipulated")
    axes[0].set_xlabel("cosine sim")

    df["topk_overlap_10pct"].hist(bins=20, ax=axes[1], edgecolor="black", color="darkorange")
    axes[1].set_title("Top-10% Jaccard overlap\nlower = more manipulated")
    axes[1].set_xlabel("Jaccard overlap")

    df["heatmap_ssim"].hist(bins=20, ax=axes[2], edgecolor="black", color="seagreen")
    axes[2].set_title("Heatmap SSIM\nlower = more manipulated")
    axes[2].set_xlabel("SSIM")

    fig.suptitle(f"Explanation-similarity distributions — {model_name}  (n={n})", fontsize=12)
    fig.tight_layout()
    path = out_dir / "distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Distributions plot → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Perceptual XAI Fragility Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, required=True,
                   help="Root of SONICS dataset")
    p.add_argument("--real-subdir", type=str, default="real",
                   help="Subdirectory name for real audio files (default: real)")
    p.add_argument("--fake-subdir", type=str, default="fake",
                   help="Subdirectory name for fake audio files (default: fake)")
    p.add_argument("--model", choices=["ast", "vggish"], default="ast")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Optional Lightning .ckpt to load fine-tuned weights")
    p.add_argument("--n-samples", type=int, default=50,
                   help="Number of test samples to attack")
    p.add_argument("--attack-steps", type=int, default=200,
                   help="Adam iterations per sample")
    p.add_argument("--attack-lr", type=float, default=1e-3)
    p.add_argument("--lambda-aud", type=float, default=1.0,
                   help="Weight of the psychoacoustic audibility loss")
    p.add_argument("--lambda-pred", type=float, default=100.0,
                   help="Weight of the prediction-preservation hinge")
    p.add_argument("--linf-bound", type=float, default=0.01,
                   help="Hard L∞ clip on the perturbation amplitude")
    p.add_argument("--clip-seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path, default=Path("reports/experiment"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds_cfg = SonicsConfig(
        root=args.data_root,
        clip_seconds=args.clip_seconds,
        real_subdir=args.real_subdir,
        fake_subdir=args.fake_subdir,
    )
    dataset = SonicsDataset(ds_cfg)
    n = min(args.n_samples, len(dataset))
    rng = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=rng)[:n].tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=(args.device.startswith("cuda")),
    )
    print(f"Dataset: {len(dataset)} total files  →  attacking {n} samples (seed={args.seed})")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = _build_model(args.model, args.checkpoint).to(args.device)
    model.eval()

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("Computing baseline metrics …")
    baseline = _baseline_metrics(model, loader, args.device)
    print(
        f"  Accuracy={baseline['accuracy']:.3f}"
        f"  AUROC={baseline['auroc']:.3f}"
        f"  EER={baseline['eer']:.3f}"
    )

    # ── Attack loop ───────────────────────────────────────────────────────────
    atk_cfg = AttackConfig(
        n_steps=args.attack_steps,
        lr=args.attack_lr,
        lambda_audibility=args.lambda_aud,
        lambda_pred=args.lambda_pred,
        linf_bound=args.linf_bound,
        log_every=20,
    )

    records: list[dict] = []
    histories: list[list[dict]] = []

    for i, (wav, label) in enumerate(tqdm(loader, desc=f"Attacking [{args.model}]")):
        wav = wav.to(args.device)

        with torch.no_grad():
            pred_orig = model(wav).argmax(-1).item()

        result = perceptual_xai_attack(model, wav, cfg=atk_cfg)

        with torch.no_grad():
            pred_adv = model(result.x_adv).argmax(-1).item()

        cos_sim = result.cosine_similarity.item()
        # topk_overlap and heatmap_ssim return [B] tensors; B=1 here.
        overlap = topk_overlap(result.cam_original, result.cam_adv, k_frac=0.1).squeeze().item()
        ssim = heatmap_ssim(result.cam_original, result.cam_adv).squeeze().item()
        d_linf = result.delta.abs().max().item()
        d_rms = result.delta.pow(2).mean().sqrt().item()
        snr = _snr_db(wav.cpu(), result.delta.cpu())

        records.append(
            {
                "sample_id": i,
                "true_label": label.item(),
                "pred_orig": pred_orig,
                "pred_adv": pred_adv,
                "prediction_preserved": bool(result.prediction_preserved.item()),
                # Explanation-similarity metrics (lower = attack worked better)
                "cosine_sim": round(cos_sim, 6),
                "topk_overlap_10pct": round(overlap, 6),
                "heatmap_ssim": round(ssim, 6),
                # Perturbation audibility metrics
                "delta_linf": round(d_linf, 8),
                "delta_rms": round(d_rms, 8),
                "snr_db": round(snr, 2) if math.isfinite(snr) else None,
            }
        )
        histories.append(result.history)

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ── Per-sample CSV ────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    csv_path = args.out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Per-sample results → {csv_path}")

    # ── Aggregate summary JSON ────────────────────────────────────────────────
    pres_rate = float(df["prediction_preserved"].mean())
    # "Success" = prediction preserved AND explanation has moved substantially.
    success_rate = float((df["prediction_preserved"] & (df["cosine_sim"] < 0.5)).mean())
    snr_vals = df["snr_db"].dropna()

    def _stats(col: str) -> dict:
        return {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
        }

    summary = {
        "model": args.model,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "n_samples": n,
        "seed": args.seed,
        "baseline": baseline,
        "attack_config": {
            "n_steps": atk_cfg.n_steps,
            "lr": atk_cfg.lr,
            "lambda_audibility": atk_cfg.lambda_audibility,
            "lambda_pred": atk_cfg.lambda_pred,
            "linf_bound": atk_cfg.linf_bound,
        },
        "results": {
            "prediction_preservation_rate": round(pres_rate, 4),
            "attack_success_rate": round(success_rate, 4),
            "cosine_sim": _stats("cosine_sim"),
            "topk_overlap_10pct": _stats("topk_overlap_10pct"),
            "heatmap_ssim": _stats("heatmap_ssim"),
            "delta_linf_mean": round(float(df["delta_linf"].mean()), 6),
            "delta_rms_mean": round(float(df["delta_rms"].mean()), 6),
            "snr_db_mean": round(float(snr_vals.mean()), 2) if len(snr_vals) else None,
        },
    }

    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        _save_convergence_plot(histories, args.model, n, args.out_dir)
    except Exception as exc:
        print(f"[warn] convergence plot failed: {exc}")

    try:
        _save_distribution_plot(df, args.model, n, args.out_dir)
    except Exception as exc:
        print(f"[warn] distributions plot failed: {exc}")

    # ── Console summary ───────────────────────────────────────────────────────
    r = summary["results"]
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  Perceptual XAI Fragility — {args.model.upper()}")
    print(sep)
    print(f"  Baseline  Acc={baseline['accuracy']:.3f}  AUROC={baseline['auroc']:.3f}  EER={baseline['eer']:.3f}")
    print(f"  Prediction preservation rate  {pres_rate:.1%}")
    print(f"  Attack success rate           {success_rate:.1%}  (preserved + cos_sim < 0.5)")
    print(f"  Cosine sim       (↓ better)  {r['cosine_sim']['mean']:.3f} ± {r['cosine_sim']['std']:.3f}"
          f"  [min={r['cosine_sim']['min']:.3f}, max={r['cosine_sim']['max']:.3f}]")
    print(f"  Top-10% Jaccard  (↓ better)  {r['topk_overlap_10pct']['mean']:.3f} ± {r['topk_overlap_10pct']['std']:.3f}")
    print(f"  Heatmap SSIM     (↓ better)  {r['heatmap_ssim']['mean']:.3f} ± {r['heatmap_ssim']['std']:.3f}")
    print(f"  Mean δ L∞                    {r['delta_linf_mean']:.5f}")
    print(f"  Mean SNR                     {r['snr_db_mean']} dB")
    print(sep)
    print(f"  All outputs → {args.out_dir}/")


if __name__ == "__main__":
    main()