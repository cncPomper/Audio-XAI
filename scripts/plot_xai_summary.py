"""Bar-chart summary of GradCAM and LRP metrics across models and attack types.

Reads sample_*.json files from the 9 standard attack run directories and
produces two figures:
  reports/gradcam_summary.png  — cos_sim and top10_overlap (GradCAM)
  reports/lrp_summary.png      — lrp_cos_sim and lrp_top10_overlap (LRP)

Each figure has two subplots (one per metric).
Bars are grouped by model (AST / VGGish / Sonics); each group has one bar
per attack type (Perceptual / PGD / X-Shift).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"
OUT_DIR   = Path(__file__).resolve().parent.parent / "reports"

RUNS = [
    ("Perceptual", "AST",    "attack/ast_epoch=4-step=9740_test_100_samples"),
    ("Perceptual", "VGGish", "attack/vggish_epoch-epoch=009_test_100_samples"),
    ("Perceptual", "Sonics", "attack/sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples"),
    ("PGD",        "AST",    "pgd_attack/pgd_ast_epoch=4-step=9740_test_100_samples"),
    ("PGD",        "VGGish", "pgd_attack/pgd_vggish_epoch-epoch=009_test_100_samples"),
    ("PGD",        "Sonics", "pgd_attack/pgd_sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples"),
    ("X-Shift",    "AST",    "xshift_attack/xshift_ast_epoch=4-step=9740_test_100_samples_shift"),
    ("X-Shift",    "VGGish", "xshift_attack/xshift_vggish_epoch-epoch=009_test_100_samples_shift"),
    ("X-Shift",    "Sonics", "xshift_attack/xshift_sonics_awsaf49_sonics-spectttra-gamma-5s_test_100_samples_shift"),
]

ATTACKS = ["Perceptual", "PGD", "X-Shift"]
MODELS  = ["AST", "VGGish", "Sonics"]

ATTACK_COLORS = {
    "Perceptual": "#4C72B0",
    "PGD":        "#DD8452",
    "X-Shift":    "#55A868",
}


def load_means() -> dict[tuple[str, str], dict[str, float | None]]:
    """Return {(attack, model): {metric: mean_or_None}}."""
    data: dict[tuple[str, str], dict[str, float | None]] = {}
    for attack, model, rel in RUNS:
        run_dir = RUNS_ROOT / rel
        samples = list(run_dir.glob("sample_*.json"))
        if not samples:
            print(f"  [warn] no samples in {rel}")
            data[(attack, model)] = {}
            continue

        records = [json.loads(p.read_text()) for p in samples]

        def mean_of(key: str) -> float | None:
            vals = [r[key] for r in records if key in r and r[key] is not None]
            return float(np.mean(vals)) if vals else None

        data[(attack, model)] = {
            "cos_sim":          mean_of("cos_sim"),
            "top10_overlap":    mean_of("top10_overlap"),
            "lrp_cos_sim":      mean_of("lrp_cos_sim"),
            "lrp_top10_overlap":mean_of("lrp_top10_overlap"),
        }
        print(f"  {attack:12s} {model:6s}  "
              f"gradcam_cos={data[(attack,model)]['cos_sim']!r:.4}  "
              f"lrp_cos={data[(attack,model)]['lrp_cos_sim']!r:.4}")
    return data


def _bar_plot(
    ax: plt.Axes,
    data: dict[tuple[str, str], dict[str, float | None]],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    n_models  = len(MODELS)
    n_attacks = len(ATTACKS)
    width     = 0.22
    x         = np.arange(n_models)

    for i, attack in enumerate(ATTACKS):
        heights = []
        for model in MODELS:
            v = data.get((attack, model), {}).get(metric)
            heights.append(v if v is not None else 0.0)

        offset = (i - n_attacks / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, heights, width,
            label=attack,
            color=ATTACK_COLORS[attack],
            edgecolor="white",
            linewidth=0.5,
        )
        # annotate with value
        for bar, h, model in zip(bars, heights, MODELS):
            if data.get((attack, model), {}).get(metric) is None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 0.01,
                    "N/A", ha="center", va="bottom", fontsize=6, color="gray",
                    rotation=90,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(
    data: dict,
    metrics: list[tuple[str, str, str]],
    out_path: Path,
    suptitle: str,
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, title, ylabel) in zip(axes, metrics):
        _bar_plot(ax, data, metric, title, ylabel)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main() -> None:
    print("Loading metrics…")
    data = load_means()

    print("\nGenerating GradCAM summary…")
    save_figure(
        data,
        metrics=[
            ("cos_sim",       "GradCAM — Cosine Similarity\n(↓ = more manipulated)",  "mean cosine sim"),
            ("top10_overlap",  "GradCAM — Top-10% Jaccard\n(↓ = more manipulated)",   "mean Jaccard overlap"),
        ],
        out_path=OUT_DIR / "gradcam_summary.png",
        suptitle="GradCAM explanation similarity — average per model & attack",
    )

    print("Generating LRP summary…")
    save_figure(
        data,
        metrics=[
            ("lrp_cos_sim",       "LRP — Cosine Similarity\n(↓ = more manipulated)",  "mean cosine sim"),
            ("lrp_top10_overlap",  "LRP — Top-10% Jaccard\n(↓ = more manipulated)",   "mean Jaccard overlap"),
        ],
        out_path=OUT_DIR / "lrp_summary.png",
        suptitle="LRP explanation similarity — average per model & attack",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()