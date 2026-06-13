from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import plot_heatmap


def plot_all_heatmaps(summary: pd.DataFrame, out_dir: str | Path, **kwargs) -> None:
    out_dir = Path(out_dir)
    bounded = dict(vmin=0, vmax=1, cmap="viridis")
    inverse = dict(vmin=0, vmax=1, cmap="viridis_r")

    plot_heatmap(
        summary,
        value_col="afs_stable_mean",
        title="AFS stable: fragility of attribution with quality and margin",
        filename=out_dir / "01_heatmap_afs_stable_model_attack.png",
        cbar_label="AFS stable, higher = more effective attack",
        **bounded,
    )
    plot_heatmap(
        summary,
        value_col="afs_quality_mean",
        title="AFS quality: fragility of attribution considering audio quality",
        filename=out_dir / "02_heatmap_afs_quality_model_attack.png",
        cbar_label="AFS quality, higher = better for attack",
        **bounded,
    )
    plot_heatmap(
        summary,
        value_col="aasr",
        title="AASR: threshold success of attribution attack",
        filename=out_dir / "03_heatmap_aasr_model_attack.png",
        cbar_label="AASR, higher = more successes",
        **bounded,
    )
    plot_heatmap(
        summary,
        value_col="pred_preserved_rate",
        title="Percentage of samples with preserved prediction",
        filename=out_dir / "04_heatmap_pred_preserved_model_attack.png",
        cbar_label="pred preserved rate",
        **bounded,
    )
    plot_heatmap(
        summary,
        value_col="quality_score_median",
        title="Normalized audio quality after attack",
        filename=out_dir / "05_heatmap_quality_model_attack.png",
        cbar_label="quality score, higher = better",
        **bounded,
    )
    plot_heatmap(
        summary,
        value_col="cos_sim_median",
        title="Median cos_sim map of attribution, lower = greater change",
        filename=out_dir / "06_heatmap_cos_sim_model_attack.png",
        cbar_label="cos_sim, lower = greater change",
        **inverse,
    )
    plot_heatmap(
        summary,
        value_col="top10_overlap_median",
        title="Median top10_overlap, lower = greater change in top regions",
        filename=out_dir / "07_heatmap_top10_overlap_model_attack.png",
        cbar_label="top10_overlap, lower = greater change in top regions",
        **inverse,
    )
    for col, title, fname in [
        (
            "afs_stable_ci_width",
            "95% CI width for AFS stable",
            "07b_heatmap_afs_stable_ci_width.png",
        ),
        ("aasr_ci_width", "95% CI width for AASR", "07c_heatmap_aasr_ci_width.png"),
    ]:
        if col in summary.columns:
            plot_heatmap(
                summary, col, title, out_dir / fname, cmap="magma", cbar_label=col
            )
