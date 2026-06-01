from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_bar_labels(
    ax, fmt: str = "{:.3f}", padding: int = 3, fontsize: int = 8
) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=padding, fontsize=fontsize)


def savefig(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def choose_text_color(value: float, vmin: float | None, vmax: float | None) -> str:
    if vmin is None or vmax is None or not np.isfinite(value) or vmax <= vmin:
        return "black"
    midpoint = vmin + 0.5 * (vmax - vmin)
    return "white" if value < midpoint else "black"


def plot_heatmap(
    summary: pd.DataFrame,
    value_col: str,
    title: str,
    filename: str | Path,
    fmt: str = ".3f",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    cbar_label: str | None = None,
    annotate: bool = True,
) -> None:
    if summary is None or summary.empty or value_col not in summary.columns:
        return

    pivot = summary.pivot(index="model", columns="attack", values=value_col)
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    fig_w = max(8, 1.15 * len(pivot.columns) + 3)
    fig_h = max(4.8, 0.65 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Delikatna siatka poprawia czytelność przy wielu modelach/atakach.
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = values[i, j]
                if pd.notna(val):
                    ax.text(
                        j,
                        i,
                        format(val, fmt),
                        ha="center",
                        va="center",
                        color=choose_text_color(val, vmin, vmax),
                        fontsize=8,
                    )
                else:
                    ax.text(
                        j, i, "—", ha="center", va="center", color="gray", fontsize=8
                    )

    ax.set_title(title)
    ax.set_xlabel("Method of attack")
    ax.set_ylabel("Model")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label or value_col)
    savefig(filename)


def plot_grouped_bar(
    pivot: pd.DataFrame,
    title: str,
    ylabel: str,
    filename: str | Path,
    ylim: Optional[tuple[float, float]] = None,
    fmt: str = "{:.3f}",
) -> None:
    if pivot is None or pivot.empty:
        return
    ax = pivot.plot(kind="bar", figsize=(max(11, len(pivot) * 1.2), 6))
    add_bar_labels(ax, fmt=fmt, fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(pivot.index.name if pivot.index.name else "")
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.xticks(rotation=0)
    savefig(filename)


def safe_filename_part(value: object) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("|", "_")
        .replace(":", "_")
    )
