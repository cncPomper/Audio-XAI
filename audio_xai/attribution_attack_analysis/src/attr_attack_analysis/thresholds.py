from __future__ import annotations

import numpy as np
import pandas as pd


def compute_threshold_sensitivity(
    df: pd.DataFrame,
    cos_values: np.ndarray | None = None,
    top10_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Liczy AASR na siatce progów cos_sim i top10_overlap.

    Pozwala sprawdzić, czy ranking ataków nie wynika wyłącznie z jednego arbitralnego
    ustawienia progów AASR.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    needed = {"model", "attack", "experiment", "pred_preserved_calc", "cos_sim", "top10_overlap"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    cos_values = cos_values if cos_values is not None else np.round(np.linspace(0.0, 1.0, 11), 2)
    top10_values = top10_values if top10_values is not None else np.round(np.linspace(0.0, 1.0, 11), 2)
    rows: list[dict[str, object]] = []
    for (model, attack, experiment), group in df.groupby(["model", "attack", "experiment"]):
        pred = group["pred_preserved_calc"].to_numpy() == 1
        cos = pd.to_numeric(group["cos_sim"], errors="coerce").to_numpy()
        top = pd.to_numeric(group["top10_overlap"], errors="coerce").to_numpy()
        valid = np.isfinite(cos) & np.isfinite(top)
        n = int(valid.sum())
        if n == 0:
            continue
        for cos_thr in cos_values:
            for top_thr in top10_values:
                success = pred & valid & (cos < cos_thr) & (top < top_thr)
                rows.append(
                    {
                        "model": model,
                        "attack": attack,
                        "experiment": experiment,
                        "cos_threshold": float(cos_thr),
                        "top10_threshold": float(top_thr),
                        "aasr": float(success.sum() / n),
                        "n_samples": n,
                    }
                )
    return pd.DataFrame(rows)
