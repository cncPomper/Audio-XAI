from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import safe_filename_part, savefig


def _plot_threshold_heatmap(subset: pd.DataFrame, title: str, filename: Path) -> None:
    pivot = subset.pivot(
        index="cos_threshold", columns="top10_threshold", values="aasr"
    ).sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([f"{x:.2f}" for x in pivot.index])
    ax.set_xlabel("top10_threshold")
    ax.set_ylabel("cos_threshold")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("AASR")
    savefig(filename)


def plot_all_threshold_plots(
    threshold_sensitivity: pd.DataFrame, out_dir: str | Path, **kwargs
) -> None:
    out_dir = Path(out_dir)
    if threshold_sensitivity is None or threshold_sensitivity.empty:
        return

    for (model, attack), subset in threshold_sensitivity.groupby(["model", "attack"]):
        safe = f"{safe_filename_part(model)}_{safe_filename_part(attack)}"
        _plot_threshold_heatmap(
            subset,
            title=f"Sensitivity of AASR to thresholds: {model} | {attack}",
            filename=out_dir / f"25_threshold_heatmap_aasr_{safe}.png",
        )

        nearest_top = subset.iloc[
            (subset["top10_threshold"] - 0.1).abs().argsort()
        ].iloc[0]["top10_threshold"]
        line_cos = subset[subset["top10_threshold"] == nearest_top].sort_values(
            "cos_threshold"
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(line_cos["cos_threshold"], line_cos["aasr"], marker="o")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("cos_threshold")
        ax.set_ylabel("AASR")
        ax.set_title(
            f"AASR vs cos_threshold at top10≈{nearest_top:.2f}\n{model} | {attack}"
        )
        ax.grid(alpha=0.25)
        savefig(out_dir / f"26_aasr_vs_cos_threshold_{safe}.png")
