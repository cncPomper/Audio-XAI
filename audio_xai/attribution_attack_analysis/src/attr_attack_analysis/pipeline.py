from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cases import make_interesting_cases
from .config import AnalysisConfig
from .filtering import filter_experiments, make_run_name
from .io import ensure_output_dirs, load_all_experiments, save_csv_outputs
from .metrics import add_all_metrics
from .plotting import plot_selected_groups
from .report import generate_markdown_report
from .summaries import (
    make_rankings,
    summarize_by_attack,
    summarize_by_model,
    summarize_by_model_attack,
    summarize_by_model_attack_label,
)
from .thresholds import compute_threshold_sensitivity


def _read_optional_csv(csv_dir: Path, filename: str) -> pd.DataFrame:
    path = csv_dir / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run_pipeline(
    config: AnalysisConfig,
    selected_models: list[str] | None = None,
    selected_attacks: list[str] | None = None,
    plot_groups: list[str] | None = None,
    save_csv: bool = True,
    make_plots: bool = True,
    only_summary: bool = False,
    only_plots: bool = False,
) -> dict[str, pd.DataFrame | Path]:
    experiments = filter_experiments(config.experiments, selected_models, selected_attacks)
    run_name = make_run_name(config.experiments, selected_models, selected_attacks)
    run_dir = config.output_dir / "runs" / run_name
    dirs = ensure_output_dirs(run_dir)

    if only_plots:
        csv_dir = dirs["csv"]
        df_all = pd.read_csv(csv_dir / "01_all_results.csv")
        summary = pd.read_csv(csv_dir / "02_summary_by_model_attack.csv")
        model_summary = pd.read_csv(csv_dir / "03_summary_by_model.csv")
        attack_summary = pd.read_csv(csv_dir / "04_summary_by_attack.csv")
        label_summary = _read_optional_csv(csv_dir, "05_summary_by_model_attack_label.csv")
        ranking = _read_optional_csv(csv_dir, "06_ranking_by_afs_stable.csv")
        threshold_sensitivity = _read_optional_csv(csv_dir, "11_threshold_sensitivity.csv")
    else:
        df_all = load_all_experiments(experiments)
        df_all = add_all_metrics(
            df_all,
            decision_threshold=config.decision_threshold,
            cos_threshold=config.cos_threshold,
            top10_threshold=config.top10_threshold,
            near_boundary_margin=config.near_boundary_margin,
            quality_thresholds=config.quality_thresholds,
        )
        summary = summarize_by_model_attack(df_all)
        model_summary = summarize_by_model(df_all)
        attack_summary = summarize_by_attack(df_all)
        label_summary = summarize_by_model_attack_label(df_all)
        ranking = make_rankings(summary)
        threshold_sensitivity = compute_threshold_sensitivity(df_all)

    interesting_cases = make_interesting_cases(df_all) if not only_plots else {}

    if save_csv and not only_plots:
        csv_frames: dict[str, pd.DataFrame] = {
            "01_all_results": df_all,
            "02_summary_by_model_attack": summary,
            "03_summary_by_model": model_summary,
            "04_summary_by_attack": attack_summary,
            "05_summary_by_model_attack_label": label_summary,
            "06_ranking_by_afs_stable": ranking,
            "11_threshold_sensitivity": threshold_sensitivity,
        }
        csv_frames.update(interesting_cases)
        save_csv_outputs(dirs["csv"], **csv_frames)

    if make_plots:
        plot_selected_groups(
            groups=plot_groups or ["all"],
            dirs=dirs,
            df=df_all,
            summary=summary,
            model_summary=model_summary,
            attack_summary=attack_summary,
            label_summary=label_summary,
            threshold=config.decision_threshold,
            only_summary=only_summary,
            threshold_sensitivity=threshold_sensitivity,
        )

    generate_markdown_report(
        summary,
        model_summary,
        attack_summary,
        dirs["run"] / "report.md",
        threshold_sensitivity=threshold_sensitivity,
    )

    print(f"\nZakończono analizę. Wyniki zapisano w: {dirs['run']}")
    print(f"CSV: {dirs['csv']}")
    print(f"Wykresy: {dirs['figures']}")
    if (dirs["interactive"] / "dashboard.html").exists():
        print(f"Dashboard HTML: {dirs['interactive'] / 'dashboard.html'}")

    return {
        "run_dir": dirs["run"],
        "df_all": df_all,
        "summary": summary,
        "model_summary": model_summary,
        "attack_summary": attack_summary,
        "label_summary": label_summary,
        "ranking": ranking,
        "threshold_sensitivity": threshold_sensitivity,
    }
