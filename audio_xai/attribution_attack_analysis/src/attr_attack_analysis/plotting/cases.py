from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import savefig


def plot_all_case_plots(df: pd.DataFrame, out_dir: str | Path, **kwargs) -> None:
    out_dir = Path(out_dir)
    if df is None or df.empty:
        return

    needed = {"afs_stable", "attribution_change", "quality_score", "margin_adv"}
    if needed.issubset(df.columns):
        top = df.sort_values("afs_stable", ascending=False).head(30).copy()
        top["case"] = top.get("_file", top.index.astype(str)).astype(str)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.scatter(
            top["quality_score"],
            top["attribution_change"],
            s=80 + 500 * top["afs_stable"],
            alpha=0.65,
        )
        for _, row in top.iterrows():
            ax.annotate(
                str(row["case"])[:24],
                (row["quality_score"], row["attribution_change"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel("Quality score")
        ax.set_ylabel("Attribution change")
        ax.set_title(
            "Top cases according to AFS stable: quality vs attribution change\npoint size = AFS stable"
        )
        ax.grid(alpha=0.25)
        savefig(out_dir / "27_top_cases_quality_vs_attr_change.png")
