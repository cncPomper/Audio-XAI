from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import savefig


def plot_all_tradeoffs(summary: pd.DataFrame, out_dir: str | Path, **kwargs) -> None:
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(9, 7))

    x = summary["pred_preserved_rate"]
    y = summary["attribution_change_mean"]
    sizes = 100 + 500 * summary["quality_score_median"].fillna(0)

    ax.scatter(x, y, s=sizes, alpha=0.65)
    for _, row in summary.iterrows():
        ax.annotate(
            f"{row['model']}\n{row['attack']}",
            xy=(row["pred_preserved_rate"], row["attribution_change_mean"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Prediction preserved rate, higher = better")
    ax.set_ylabel("Mean attribution change, higher = more change")
    ax.set_title(
        "Trade-off: prediction preservation vs attribution change\npoint size = audio quality"
    )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, max(1.0, y.max() * 1.15))
    savefig(out_dir / "12_tradeoff_pred_preserved_vs_attr_change_quality.png")
