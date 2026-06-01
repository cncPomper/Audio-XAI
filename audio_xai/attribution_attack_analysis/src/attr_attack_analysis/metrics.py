from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "pred_orig",
    "pred_adv",
    "prob_orig",
    "prob_adv",
    "cos_sim",
    "top10_overlap",
]


def validate_required_columns(
    df: pd.DataFrame, required: list[str] | None = None
) -> None:
    required = required or REQUIRED_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Brakuje wymaganych kolumn w danych: {missing}")


def normalize_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    quality_norm_cols: list[str] = []

    if "pesq" in df.columns:
        df["pesq_norm"] = ((df["pesq"] - 1.0) / (4.5 - 1.0)).clip(0, 1)
        quality_norm_cols.append("pesq_norm")
    if "stoi" in df.columns:
        df["stoi_norm"] = df["stoi"].clip(0, 1)
        quality_norm_cols.append("stoi_norm")
    if "visqol" in df.columns:
        df["visqol_norm"] = ((df["visqol"] - 1.0) / (5.0 - 1.0)).clip(0, 1)
        quality_norm_cols.append("visqol_norm")
    if "peaq" in df.columns:
        df["peaq_norm"] = ((df["peaq"] + 4.0) / 4.0).clip(0, 1)
        quality_norm_cols.append("peaq_norm")
    if "zimtohrli" in df.columns:
        df["zimtohrli_norm"] = (df["zimtohrli"] / 5.0).clip(0, 1)
        quality_norm_cols.append("zimtohrli_norm")
    if "cdpam" in df.columns:
        df["cdpam_norm"] = 1 - df["cdpam"].clip(0, 1)
        # print(f"CDPAM znormalizowane do {df['cdpam_norm']}")
        quality_norm_cols.append("cdpam_norm")

    if quality_norm_cols:
        df["quality_score"] = df[quality_norm_cols].mean(axis=1)
    else:
        df["quality_score"] = 1.0
    return df


def add_prediction_metrics(
    df: pd.DataFrame, decision_threshold: float = 0.5, near_boundary_margin: float = 0.1
) -> pd.DataFrame:
    df = df.copy()
    validate_required_columns(df, ["pred_orig", "pred_adv", "prob_orig", "prob_adv"])

    df["pred_preserved_calc"] = (df["pred_orig"] == df["pred_adv"]).astype(int)
    df["score_shift"] = df["prob_adv"] - df["prob_orig"]
    df["abs_score_shift"] = df["score_shift"].abs()

    df["margin_orig"] = (df["prob_orig"] - decision_threshold).abs()
    df["margin_adv"] = (df["prob_adv"] - decision_threshold).abs()
    df["margin_drop"] = df["margin_orig"] - df["margin_adv"]

    denom = max(decision_threshold, 1e-12)
    df["margin_safety"] = (df["margin_adv"] / denom).clip(0, 1)

    df["near_boundary_adv_005"] = (df["margin_adv"] < 0.05).astype(int)
    df["near_boundary_adv_010"] = (df["margin_adv"] < 0.10).astype(int)
    df["near_boundary_adv_custom"] = (df["margin_adv"] < near_boundary_margin).astype(
        int
    )

    return df


def add_attribution_metrics(
    df: pd.DataFrame, cos_threshold: float = 0.2, top10_threshold: float = 0.1
) -> pd.DataFrame:
    df = df.copy()
    validate_required_columns(df, ["cos_sim", "top10_overlap"])

    if df["cos_sim"].min(skipna=True) < 0:
        df["cos_sim_norm"] = ((df["cos_sim"] + 1.0) / 2.0).clip(0, 1)
    else:
        df["cos_sim_norm"] = df["cos_sim"].clip(0, 1)

    df["top10_overlap_norm"] = df["top10_overlap"].clip(0, 1)
    df["cos_change"] = 1.0 - df["cos_sim_norm"]
    df["top10_change"] = 1.0 - df["top10_overlap_norm"]
    df["attribution_change"] = 0.5 * (df["cos_change"] + df["top10_change"])

    df["attr_changed_cos"] = (df["cos_sim"] < cos_threshold).astype(int)
    df["attr_changed_top10"] = (df["top10_overlap"] < top10_threshold).astype(int)
    df["attr_changed_either"] = (
        (df["attr_changed_cos"] == 1) | (df["attr_changed_top10"] == 1)
    ).astype(int)
    df["attr_changed_both"] = (
        (df["attr_changed_cos"] == 1) & (df["attr_changed_top10"] == 1)
    ).astype(int)

    return df


def add_success_metrics(
    df: pd.DataFrame, quality_thresholds: dict[str, float] | None = None
) -> pd.DataFrame:
    df = df.copy()
    quality_thresholds = quality_thresholds or {
        "pesq": 3.0,
        "stoi": 0.90,
        "visqol": 3.5,
    }

    needed = [
        "pred_preserved_calc",
        "attr_changed_cos",
        "attr_changed_top10",
        "attr_changed_either",
        "attr_changed_both",
    ]
    validate_required_columns(df, needed)

    df["aasr_cos_only"] = (
        (df["pred_preserved_calc"] == 1) & (df["attr_changed_cos"] == 1)
    ).astype(int)
    df["aasr_top10_only"] = (
        (df["pred_preserved_calc"] == 1) & (df["attr_changed_top10"] == 1)
    ).astype(int)
    df["aasr_either"] = (
        (df["pred_preserved_calc"] == 1) & (df["attr_changed_either"] == 1)
    ).astype(int)
    df["aasr_strict"] = (
        (df["pred_preserved_calc"] == 1) & (df["attr_changed_both"] == 1)
    ).astype(int)
    df["attribution_attack_success"] = df["aasr_strict"]

    quality_condition = pd.Series(True, index=df.index)
    for metric, min_value in quality_thresholds.items():
        if metric in df.columns:
            quality_condition &= df[metric] >= min_value
    df["quality_pass"] = quality_condition.astype(int)
    df["aasr_quality"] = (
        (df["attribution_attack_success"] == 1) & (df["quality_pass"] == 1)
    ).astype(int)
    return df


def add_fragility_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    validate_required_columns(
        df,
        ["pred_preserved_calc", "attribution_change", "quality_score", "margin_safety"],
    )

    df["afs_simple"] = df["pred_preserved_calc"] * df["attribution_change"]
    df["afs_quality"] = df["afs_simple"] * df["quality_score"]
    df["afs_stable"] = df["afs_quality"] * df["margin_safety"]
    return df


def add_diagnostic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "correct_orig" in df.columns and "correct_adv" in df.columns:
        df["accuracy_drop_sample"] = df["correct_orig"] - df["correct_adv"]
    return df


def add_all_metrics(
    df: pd.DataFrame,
    decision_threshold: float = 0.5,
    cos_threshold: float = 0.2,
    top10_threshold: float = 0.1,
    near_boundary_margin: float = 0.1,
    quality_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    validate_required_columns(df)
    df = add_prediction_metrics(df, decision_threshold, near_boundary_margin)
    df = add_attribution_metrics(df, cos_threshold, top10_threshold)
    df = normalize_quality_scores(df)
    df = add_success_metrics(df, quality_thresholds)
    df = add_fragility_metrics(df)
    df = add_diagnostic_metrics(df)
    return df
