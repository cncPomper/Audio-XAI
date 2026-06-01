from __future__ import annotations

import pandas as pd


def _select_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "model", "attack", "experiment", "_file", "label_str", "pred_orig", "pred_adv",
        "prob_orig", "prob_adv", "score_shift", "abs_score_shift", "cos_sim", "top10_overlap",
        "attribution_change", "afs_stable", "quality_score", "margin_adv", "margin_safety",
        "aasr_strict", "aasr_quality", "quality_pass", "near_boundary_adv_010",
        "pesq", "stoi", "visqol", "peaq", "zimtohrli",
    ]
    return [c for c in preferred if c in df.columns]


def make_interesting_cases(df: pd.DataFrame, top_n: int = 50) -> dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return {}
    cols = _select_columns(df)
    data = df.copy()
    result: dict[str, pd.DataFrame] = {}

    result["07_top_success_cases"] = (
        data[(data.get("pred_preserved_calc", 0) == 1) & (data.get("quality_pass", 1) == 1)]
        .sort_values(["afs_stable", "attribution_change", "quality_score"], ascending=[False, False, False])
        .head(top_n)[cols]
    )
    result["08_top_attribution_change_cases"] = (
        data.sort_values(["attribution_change", "afs_stable"], ascending=[False, False]).head(top_n)[cols]
    )
    result["09_boundary_cases"] = (
        data.sort_values(["margin_adv", "afs_stable"], ascending=[True, False]).head(top_n)[cols]
    )
    if "quality_score" in data.columns:
        result["10_quality_failures"] = (
            data[(data.get("aasr_strict", 0) == 1) & (data.get("quality_pass", 1) == 0)]
            .sort_values(["quality_score", "afs_stable"], ascending=[True, False])
            .head(top_n)[cols]
        )
    return {k: v for k, v in result.items() if v is not None and not v.empty}
