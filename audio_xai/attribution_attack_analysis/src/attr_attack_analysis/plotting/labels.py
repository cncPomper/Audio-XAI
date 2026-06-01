from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import plot_heatmap, safe_filename_part


def plot_all_label_plots(
    label_summary: pd.DataFrame, out_dir: str | Path, **kwargs
) -> None:
    out_dir = Path(out_dir)
    if label_summary is None or label_summary.empty:
        print("Brak label_summary albo kolumny label_str — pomijam wykresy fake/real.")
        return

    for label in sorted(label_summary["label_str"].dropna().unique()):
        subset = label_summary[label_summary["label_str"] == label]
        safe_label = safe_filename_part(label)
        plot_heatmap(
            subset,
            value_col="afs_stable_mean",
            title=f"AFS stable for class {label}",
            filename=out_dir / f"14_heatmap_afs_stable_label_{safe_label}.png",
        )
        plot_heatmap(
            subset,
            value_col="aasr",
            title=f"AASR for class {label}",
            filename=out_dir / f"15_heatmap_aasr_label_{safe_label}.png",
        )
