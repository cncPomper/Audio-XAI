from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import savefig


def plot_all_diagnostics(summary: pd.DataFrame, out_dir: str | Path, **kwargs) -> None:
    out_dir = Path(out_dir)
    if summary is None or summary.empty:
        return

    diag_cols = [
        "pred_preserved_rate",
        "attr_changed_cos_rate",
        "attr_changed_top10_rate",
        "attr_changed_both_rate",
        "aasr_cos_only",
        "aasr_top10_only",
        "aasr_either",
        "aasr",
        "aasr_quality",
    ]
    diag_cols = [c for c in diag_cols if c in summary.columns]
    diag = summary.copy()
    diag["combo"] = diag["model"] + "\n" + diag["attack"]
    diag_pivot = diag.set_index("combo")[diag_cols]

    ax = diag_pivot.plot(kind="bar", figsize=(14, 7))
    ax.set_title("Diagnostics of AASR conditions")
    ax.set_ylabel("Proportion of samples")
    ax.set_xlabel("Model x Attack method")
    ax.set_ylim(0, 1.08)
    plt.xticks(rotation=0)
    savefig(out_dir / "13_aasr_diagnostics_conditions.png")

    bottleneck_cols = {
        "pred_preserved_rate",
        "attr_changed_cos_rate",
        "attr_changed_top10_rate",
        "aasr",
        "aasr_quality",
    }
    if bottleneck_cols.issubset(summary.columns):
        b = summary.copy()
        b["combo"] = b["model"] + "\n" + b["attack"]
        # Szacunkowy funnel: co odpada po kolejnych warunkach. To nie jest dokładny rozkład rozłączny,
        # ale dobrze wskazuje, który warunek jest główną barierą sukcesu.
        b["prediction_changed_or_unstable"] = 1 - b["pred_preserved_rate"]
        b["attribution_not_changed_enough"] = (
            b["pred_preserved_rate"] - b["aasr"]
        ).clip(lower=0)
        b["quality_failed_after_aasr"] = (b["aasr"] - b["aasr_quality"]).clip(lower=0)
        b["full_quality_success"] = b["aasr_quality"]
        cols = [
            "prediction_changed_or_unstable",
            "attribution_not_changed_enough",
            "quality_failed_after_aasr",
            "full_quality_success",
        ]
        ax = b.set_index("combo")[cols].plot(kind="bar", stacked=True, figsize=(14, 7))
        ax.set_title("Funnel / bottleneck AASR: where samples drop out")
        ax.set_ylabel("Proportion of samples")
        ax.set_xlabel("Model x Attack method")
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=0)
        savefig(out_dir / "13b_aasr_bottleneck_stacked.png")
