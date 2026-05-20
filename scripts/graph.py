#!/usr/bin/env python3
"""
Read experiment results from an --out-dir produced by run_experiment.py,
print the console summary, and regenerate the distributions plot.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_distribution_plot(df: pd.DataFrame, model_name: str, n: int, out_dir: Path) -> None:
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Display results from a run_experiment.py output directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/experiment"),
        help="Directory produced by run_experiment.py",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip regenerating the distributions plot",
    )
    args = p.parse_args()

    summary_path = args.out_dir / "summary.json"
    csv_path = args.out_dir / "results.csv"

    if not summary_path.exists():
        sys.exit(f"ERROR: {summary_path} not found. Check --out-dir.")
    if not csv_path.exists():
        sys.exit(f"ERROR: {csv_path} not found. Check --out-dir.")

    with open(summary_path) as f:
        summary = json.load(f)

    df = pd.read_csv(csv_path)

    model = summary["model"]
    n = summary["n_samples"]
    baseline = summary["baseline"]
    r = summary["results"]

    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  Perceptual XAI Fragility — {model.upper()}")
    print(sep)
    print(f"  Samples:  {n}   Seed: {summary['seed']}")
    print(f"  Checkpoint: {summary['checkpoint'] or 'pretrained'}")
    print(sep)
    print(f"  Baseline  Acc={baseline['accuracy']:.3f}  AUROC={baseline['auroc']:.3f}  EER={baseline['eer']:.3f}")
    print(f"  Prediction preservation rate  {r['prediction_preservation_rate']:.1%}")
    print(f"  Attack success rate           {r['attack_success_rate']:.1%}  (preserved + cos_sim < 0.5)")
    print(f"  Cosine sim       (↓ better)  {r['cosine_sim']['mean']:.3f} ± {r['cosine_sim']['std']:.3f}"
          f"  [min={r['cosine_sim']['min']:.3f}, max={r['cosine_sim']['max']:.3f}]")
    print(f"  Top-10% Jaccard  (↓ better)  {r['topk_overlap_10pct']['mean']:.3f} ± {r['topk_overlap_10pct']['std']:.3f}")
    print(f"  Heatmap SSIM     (↓ better)  {r['heatmap_ssim']['mean']:.3f} ± {r['heatmap_ssim']['std']:.3f}")
    print(f"  Mean δ L∞                    {r['delta_linf_mean']:.5f}")
    print(f"  Mean SNR                     {r['snr_db_mean']} dB")
    print(sep)
    print(f"  All outputs → {args.out_dir}/")

    if not args.no_plot:
        _save_distribution_plot(df, model, n, args.out_dir)


if __name__ == "__main__":
    main()