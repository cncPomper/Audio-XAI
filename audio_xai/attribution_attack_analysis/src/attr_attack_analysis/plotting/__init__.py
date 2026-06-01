from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .heatmaps import plot_all_heatmaps
from .rankings import plot_all_rankings
from .tradeoffs import plot_all_tradeoffs
from .diagnostics import plot_all_diagnostics
from .labels import plot_all_label_plots
from .audio import plot_all_audio_plots
from .scores import plot_all_score_plots
from .distributions import plot_all_distribution_plots
from .thresholds import plot_all_threshold_plots
from .cases import plot_all_case_plots
from .interactive import plot_all_interactive_plots


PLOTTERS = {
    "heatmaps": plot_all_heatmaps,
    "rankings": plot_all_rankings,
    "tradeoffs": plot_all_tradeoffs,
    "diagnostics": plot_all_diagnostics,
    "labels": plot_all_label_plots,
    "audio": plot_all_audio_plots,
    "scores": plot_all_score_plots,
    "distributions": plot_all_distribution_plots,
    "thresholds": plot_all_threshold_plots,
    "cases": plot_all_case_plots,
    "interactive": plot_all_interactive_plots,
}


def resolve_plot_groups(groups: Iterable[str] | None) -> list[str]:
    groups = list(groups or ["all"])
    if "all" in groups:
        return list(PLOTTERS.keys())
    unknown = sorted(set(groups) - set(PLOTTERS))
    if unknown:
        raise ValueError(f"Nieznane grupy wykresów: {unknown}. Dostępne: {sorted(PLOTTERS)} lub all")
    return groups


def plot_selected_groups(
    groups: Iterable[str] | None,
    dirs: dict[str, Path],
    df: pd.DataFrame,
    summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    attack_summary: pd.DataFrame,
    label_summary: pd.DataFrame,
    threshold: float,
    only_summary: bool = False,
    threshold_sensitivity: pd.DataFrame | None = None,
) -> None:
    selected = resolve_plot_groups(groups)
    if only_summary:
        selected = [g for g in selected if g not in {"scores", "distributions", "cases", "interactive"}]

    for group in selected:
        out_dir = dirs.get(group, dirs["figures"])
        if group == "rankings":
            PLOTTERS[group](summary=summary, model_summary=model_summary, attack_summary=attack_summary, out_dir=out_dir)
        elif group == "labels":
            PLOTTERS[group](label_summary=label_summary, out_dir=out_dir)
        elif group == "scores":
            PLOTTERS[group](df=df, out_dir=out_dir, threshold=threshold)
        elif group == "distributions":
            PLOTTERS[group](df=df, out_dir=out_dir)
        elif group == "thresholds":
            PLOTTERS[group](threshold_sensitivity=threshold_sensitivity, out_dir=out_dir)
        elif group == "cases":
            PLOTTERS[group](df=df, out_dir=out_dir)
        elif group == "interactive":
            PLOTTERS[group](df=df, summary=summary, threshold_sensitivity=threshold_sensitivity, out_dir=out_dir)
        else:
            PLOTTERS[group](summary=summary, out_dir=out_dir)
