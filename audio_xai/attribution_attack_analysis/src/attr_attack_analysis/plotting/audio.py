from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import add_bar_labels, savefig


def plot_all_audio_plots(summary: pd.DataFrame, out_dir: str | Path, **kwargs) -> None:
    out_dir = Path(out_dir)
    audio_cols = [
        "pesq_median",
        "stoi_median",
        "visqol_median",
        "peaq_median",
        "zimtohrli_median",
        "cdpam_median",
    ]
    audio_cols = [c for c in audio_cols if c in summary.columns]
    if not audio_cols:
        return

    audio = summary.copy()
    audio["combo"] = audio["model"] + "\n" + audio["attack"]
    audio_pivot = audio.set_index("combo")[audio_cols]

    audio_pivot.to_csv(out_dir / "99_audio_quality_model_attack.csv")

    ax = audio_pivot.plot(kind="bar", figsize=(14, 7))
    add_bar_labels(ax, fmt="{:.3f}", fontsize=7)
    ax.set_title("Audio quality after attack for combinations model x attack")
    ax.set_ylabel("Median values")
    ax.set_xlabel("Model x attack method")

    description = (
        "PESQ [-0.5-4.5] higher = better; "
        "STOI [0-1] higher = better; "
        "ViSQOL [1-5] higher = better; "
        "PEAQ [-4-0] closer to 0 = better; "
        "Zimtohrli [0-5] higher = better; "
        "CDPAM similarity [0-1] higher = better."
    )
    ax.text(
        0.5,
        -0.24,
        description,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )
    plt.xticks(rotation=0)
    savefig(out_dir / "16_audio_quality_model_attack.png")
