from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import savefig


def _combo(df: pd.DataFrame) -> pd.Series:
    return df["model"].astype(str) + "\n" + df["attack"].astype(str)


def _boxplot_metric(
    df: pd.DataFrame, metric: str, title: str, ylabel: str, out_path: Path
) -> None:
    if metric not in df.columns or df.empty:
        return
    tmp = df[["model", "attack", metric]].dropna().copy()
    if tmp.empty:
        return
    tmp["combo"] = _combo(tmp)
    labels = sorted(tmp["combo"].unique())
    values = [tmp.loc[tmp["combo"] == label, metric].to_numpy() for label in labels]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
    ax.boxplot(values, labels=labels, showmeans=True)
    # Jitterowane punkty pokazują liczebność i outliery bez ukrywania rozkładu.
    rng = np.random.default_rng(7)
    for i, vals in enumerate(values, start=1):
        if len(vals) == 0:
            continue
        x = rng.normal(i, 0.035, size=len(vals))
        ax.scatter(x, vals, alpha=0.25, s=12)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model x attack method")
    plt.xticks(rotation=0)
    savefig(out_path)


def _ecdf_metric(
    df: pd.DataFrame, metric: str, title: str, xlabel: str, out_path: Path
) -> None:
    if metric not in df.columns or df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for attack, subset in df.groupby("attack"):
        values = (
            pd.to_numeric(subset[metric], errors="coerce")
            .dropna()
            .sort_values()
            .to_numpy()
        )
        if len(values) == 0:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", label=str(attack))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF: percentage of samples below value")
    ax.grid(alpha=0.25)
    ax.legend(title="Attack")
    savefig(out_path)


def plot_all_distribution_plots(
    df: pd.DataFrame, out_dir: str | Path, **kwargs
) -> None:
    out_dir = Path(out_dir)
    if df is None or df.empty:
        return

    specs = [
        (
            "afs_stable",
            "Distribution of AFS stable for model x attack pairs",
            "AFS stable",
            "18_box_afs_stable.png",
        ),
        (
            "attribution_change",
            "Distribution of attribution change for model x attack pairs",
            "Attribution change",
            "19_box_attribution_change.png",
        ),
        (
            "quality_score",
            "Distribution of audio quality for model x attack pairs",
            "Quality score",
            "20_box_quality_score.png",
        ),
        (
            "margin_adv",
            "Distribution of margin after attack for model x attack pairs",
            "Margin adv",
            "21_box_margin_adv.png",
        ),
    ]
    for metric, title, ylabel, filename in specs:
        _boxplot_metric(df, metric, title, ylabel, out_dir / filename)

    _ecdf_metric(
        df,
        "cos_sim",
        "ECDF cos_sim by attack methods — lower values indicate greater change",
        "cos_sim",
        out_dir / "22_ecdf_cos_sim_by_attack.png",
    )
    _ecdf_metric(
        df,
        "top10_overlap",
        "ECDF top10_overlap by attack methods — lower values indicate greater change in top regions",
        "top10_overlap",
        out_dir / "23_ecdf_top10_overlap_by_attack.png",
    )

    if {"cos_sim", "top10_overlap", "afs_stable"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            df["cos_sim"], df["top10_overlap"], c=df["afs_stable"], alpha=0.65, s=35
        )
        ax.set_xlabel("cos_sim, lower = greater change")
        ax.set_ylabel("top10_overlap, lower = greater change in top-k")
        ax.set_title("Sample map: cos_sim vs top10_overlap, color = AFS stable")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("AFS stable")
        savefig(out_dir / "24_scatter_cos_vs_top10_afs_stable.png")
