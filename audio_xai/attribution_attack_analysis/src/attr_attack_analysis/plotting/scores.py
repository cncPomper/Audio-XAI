from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import safe_filename_part, savefig


def plot_all_score_plots(
    df: pd.DataFrame, out_dir: str | Path, threshold: float = 0.5, **kwargs
) -> None:
    out_dir = Path(out_dir)
    if df is None or df.empty:
        return

    for (model, attack), subset in df.groupby(["model", "attack"]):
        fig, ax = plt.subplots(figsize=(7, 6))

        color_col = "afs_stable" if "afs_stable" in subset.columns else None
        size_col = "quality_score" if "quality_score" in subset.columns else None
        sizes = 35
        if size_col:
            sizes = 25 + 130 * subset[size_col].fillna(0)
        if color_col:
            sc = ax.scatter(
                subset["prob_orig"],
                subset["prob_adv"],
                c=subset[color_col],
                s=sizes,
                alpha=0.7,
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("AFS stable")
        elif "label_str" in subset.columns:
            for label in sorted(subset["label_str"].dropna().unique()):
                s = subset[subset["label_str"] == label]
                ax.scatter(s["prob_orig"], s["prob_adv"], alpha=0.6, label=label)
            ax.legend()
        else:
            ax.scatter(subset["prob_orig"], subset["prob_adv"], alpha=0.6)

        if "aasr_quality" in subset.columns:
            success = subset[subset["aasr_quality"] == 1]
            if not success.empty:
                ax.scatter(
                    success["prob_orig"],
                    success["prob_adv"],
                    facecolors="none",
                    edgecolors="black",
                    s=170,
                    linewidths=1.2,
                    label="AASR+quality",
                )
                ax.legend()

        ax.plot([0, 1], [0, 1], linestyle="--", label="bez zmiany")
        ax.axvline(threshold, linestyle=":", linewidth=1)
        ax.axhline(threshold, linestyle=":", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Fake score before attack")
        ax.set_ylabel("Fake score after attack")
        ax.set_title(f"Fake score before and after attack\n{model} | {attack}")

        description = (
            f"Decision threshold = {threshold}. Color = AFS stable, size = audio quality; "
            "black edges indicate full AASR success with preserved quality."
        )
        ax.text(
            0.5,
            -0.18,
            description,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            wrap=True,
        )
        savefig(
            out_dir
            / f"17_prob_orig_vs_prob_adv_{safe_filename_part(model)}_{safe_filename_part(attack)}.png"
        )
