from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import plot_grouped_bar, savefig
from ..constants import SUMMARY_COLUMNS_FOR_AGGREGATES


def _errorbar_ranking(
    summary: pd.DataFrame, metric: str, title: str, ylabel: str, filename: Path
) -> None:
    if metric not in summary.columns or summary.empty:
        return
    ranking = summary.sort_values(metric, ascending=False).copy()
    ranking["combo"] = ranking["model"] + "\n" + ranking["attack"]
    fig, ax = plt.subplots(figsize=(max(12, len(ranking) * 0.9), 6))
    y = ranking[metric].to_numpy(dtype=float)
    x = np.arange(len(ranking))
    # bars = ax.bar(x, y)
    # low_col = metric.replace("_mean", "") + "_ci_low"
    # high_col = metric.replace("_mean", "") + "_ci_high"
    # if low_col in ranking.columns and high_col in ranking.columns:
    #     low = ranking[low_col].to_numpy(dtype=float)
    #     high = ranking[high_col].to_numpy(dtype=float)
    #     yerr = np.vstack([np.maximum(0, y - low), np.maximum(0, high - y)])
    #     ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=4, linewidth=1)
    # ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    bars = ax.bar(x, y)
    label_tops = y.copy()
    low_col = metric.replace("_mean", "") + "_ci_low"
    high_col = metric.replace("_mean", "") + "_ci_high"
    if low_col in ranking.columns and high_col in ranking.columns:
        low = ranking[low_col].to_numpy(dtype=float)
        high = ranking[high_col].to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(0, y - low), np.maximum(0, high - y)])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            capsize=3,
            color="black",
            linewidth=2,
            capthick=1,
        )
        label_tops = y + yerr[1]
    for i, (val, top) in enumerate(zip(y, label_tops)):
        ax.text(i, top + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ranking["combo"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model x Method of Attack")
    ax.set_ylim(0, max(1.0, np.nanmax(y) * 1.2 if len(y) else 1.0))
    plt.xticks(rotation=0)
    savefig(filename)


def plot_all_rankings(
    summary: pd.DataFrame,
    model_summary: pd.DataFrame | None = None,
    attack_summary: pd.DataFrame | None = None,
    out_dir: str | Path = ".",
    **kwargs,
) -> None:
    out_dir = Path(out_dir)

    _errorbar_ranking(
        summary,
        metric="afs_stable_mean",
        title="Ranking of model x attack combinations by AFS stable with 95% CI",
        ylabel="AFS stable, higher = greater attribution fragility",
        filename=out_dir / "08_ranking_model_attack_afs_stable.png",
    )

    ranking_aasr = summary.sort_values("aasr", ascending=False).copy()
    ranking_aasr["combo"] = ranking_aasr["model"] + "\n" + ranking_aasr["attack"]
    fig, ax = plt.subplots(figsize=(max(12, len(ranking_aasr) * 0.9), 6))
    x = np.arange(len(ranking_aasr))
    y = ranking_aasr["aasr"].to_numpy(dtype=float)
    bars = ax.bar(x, y)
    label_tops = y.copy()
    if {"aasr_ci_low", "aasr_ci_high"}.issubset(ranking_aasr.columns):
        low = ranking_aasr["aasr_ci_low"].to_numpy(dtype=float)
        high = ranking_aasr["aasr_ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(0, y - low), np.maximum(0, high - y)])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            capsize=2,
            color="black",
            linewidth=2,
            capthick=1,
        )
        label_tops = y + yerr[1]
    for i, (val, top) in enumerate(zip(y, label_tops)):
        ax.text(i, top + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ranking_aasr["combo"])
    ax.set_title("Ranking of model x attack combinations by AASR with 95% CI")
    ax.set_ylabel("AASR, higher = more threshold successes")
    ax.set_xlabel("Model x Attack method")
    ax.set_ylim(0, max(1.0, np.nanmax(y) * 1.15 if len(y) else 1.0))
    plt.xticks(rotation=0)
    savefig(out_dir / "09_ranking_model_attack_aasr.png")

    if model_summary is not None and not model_summary.empty:
        cols = [c for c in SUMMARY_COLUMNS_FOR_AGGREGATES if c in model_summary.columns]
        model_pivot = model_summary.set_index("model")[cols]
        plot_grouped_bar(
            model_pivot,
            title="Comparison of models, averaged over attack methods",
            ylabel="Value",
            filename=out_dir / "10_model_aggregate_comparison.png",
            ylim=(0, max(1.0, model_pivot.max().max() * 1.15)),
        )

    if attack_summary is not None and not attack_summary.empty:
        cols = [
            c for c in SUMMARY_COLUMNS_FOR_AGGREGATES if c in attack_summary.columns
        ]
        attack_pivot = attack_summary.set_index("attack")[cols]
        plot_grouped_bar(
            attack_pivot,
            title="Comparison of attack methods, averaged over models",
            ylabel="Value",
            filename=out_dir / "11_attack_aggregate_comparison.png",
            ylim=(0, max(1.0, attack_pivot.max().max() * 1.15)),
        )
