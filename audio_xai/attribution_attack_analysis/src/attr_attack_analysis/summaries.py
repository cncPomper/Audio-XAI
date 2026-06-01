from __future__ import annotations

import numpy as np
import pandas as pd


def _optional_aggs(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    optional_specs = {
        "accuracy_orig": ("correct_orig", "mean"),
        "accuracy_adv": ("correct_adv", "mean"),
        "pesq_median": ("pesq", "median"),
        "stoi_median": ("stoi", "median"),
        "visqol_median": ("visqol", "median"),
        "peaq_median": ("peaq", "median"),
        "zimtohrli_median": ("zimtohrli", "median"),
        "cdpam_median": ("cdpam_norm", "median"),
    }
    return {out: spec for out, spec in optional_specs.items() if spec[0] in df.columns}


def _mean_ci(series: pd.Series, z: float = 1.96) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    return max(0.0, mean - z * sem), min(1.0, mean + z * sem)


def _add_ci_columns(
    summary: pd.DataFrame, df: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    metric_cols = [
        "afs_stable",
        "afs_quality",
        "afs_simple",
        "aasr_strict",
        "aasr_quality",
        "attribution_change",
    ]
    present = [c for c in metric_cols if c in df.columns]
    if not present or summary.empty:
        return summary

    rows: list[dict[str, object]] = []
    for key, group in df.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        for metric in present:
            low, high = _mean_ci(group[metric])
            out_name = "aasr" if metric == "aasr_strict" else metric
            row[f"{out_name}_ci_low"] = low
            row[f"{out_name}_ci_high"] = high
            row[f"{out_name}_ci_width"] = (
                high - low if pd.notna(low) and pd.notna(high) else np.nan
            )
        rows.append(row)
    ci = pd.DataFrame(rows)
    return summary.merge(ci, on=group_cols, how="left")


def summarize_by_model_attack(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "n_samples": ("experiment", "size"),
        "pred_preserved_rate": ("pred_preserved_calc", "mean"),
        "attr_changed_cos_rate": ("attr_changed_cos", "mean"),
        "attr_changed_top10_rate": ("attr_changed_top10", "mean"),
        "attr_changed_both_rate": ("attr_changed_both", "mean"),
        "aasr_cos_only": ("aasr_cos_only", "mean"),
        "aasr_top10_only": ("aasr_top10_only", "mean"),
        "aasr_either": ("aasr_either", "mean"),
        "aasr": ("aasr_strict", "mean"),
        "aasr_quality": ("aasr_quality", "mean"),
        "afs_simple_mean": ("afs_simple", "mean"),
        "afs_simple_median": ("afs_simple", "median"),
        "afs_simple_std": ("afs_simple", "std"),
        "afs_simple_p25": ("afs_simple", lambda x: x.quantile(0.25)),
        "afs_simple_p75": ("afs_simple", lambda x: x.quantile(0.75)),
        "afs_quality_mean": ("afs_quality", "mean"),
        "afs_quality_median": ("afs_quality", "median"),
        "afs_quality_std": ("afs_quality", "std"),
        "afs_quality_p25": ("afs_quality", lambda x: x.quantile(0.25)),
        "afs_quality_p75": ("afs_quality", lambda x: x.quantile(0.75)),
        "afs_stable_mean": ("afs_stable", "mean"),
        "afs_stable_median": ("afs_stable", "median"),
        "afs_stable_std": ("afs_stable", "std"),
        "afs_stable_p25": ("afs_stable", lambda x: x.quantile(0.25)),
        "afs_stable_p75": ("afs_stable", lambda x: x.quantile(0.75)),
        "attribution_change_mean": ("attribution_change", "mean"),
        "attribution_change_median": ("attribution_change", "median"),
        "attribution_change_p25": ("attribution_change", lambda x: x.quantile(0.25)),
        "attribution_change_p75": ("attribution_change", lambda x: x.quantile(0.75)),
        "cos_sim_mean": ("cos_sim", "mean"),
        "cos_sim_median": ("cos_sim", "median"),
        "cos_sim_p25": ("cos_sim", lambda x: x.quantile(0.25)),
        "cos_sim_p75": ("cos_sim", lambda x: x.quantile(0.75)),
        "top10_overlap_mean": ("top10_overlap", "mean"),
        "top10_overlap_median": ("top10_overlap", "median"),
        "top10_overlap_p25": ("top10_overlap", lambda x: x.quantile(0.25)),
        "top10_overlap_p75": ("top10_overlap", lambda x: x.quantile(0.75)),
        "abs_score_shift_mean": ("abs_score_shift", "mean"),
        "abs_score_shift_median": ("abs_score_shift", "median"),
        "margin_adv_mean": ("margin_adv", "mean"),
        "margin_adv_median": ("margin_adv", "median"),
        "margin_drop_mean": ("margin_drop", "mean"),
        "margin_drop_median": ("margin_drop", "median"),
        "near_boundary_adv_005": ("near_boundary_adv_005", "mean"),
        "near_boundary_adv_010": ("near_boundary_adv_010", "mean"),
        "near_boundary_adv_custom": ("near_boundary_adv_custom", "mean"),
        "quality_score_mean": ("quality_score", "mean"),
        "quality_score_median": ("quality_score", "median"),
        "quality_score_p25": ("quality_score", lambda x: x.quantile(0.25)),
        "quality_score_p75": ("quality_score", lambda x: x.quantile(0.75)),
    }
    agg_dict.update(_optional_aggs(df))

    summary = (
        df.groupby(["model", "attack", "experiment"]).agg(**agg_dict).reset_index()
    )
    summary = _add_ci_columns(summary, df, ["model", "attack", "experiment"])
    if "accuracy_orig" in summary.columns and "accuracy_adv" in summary.columns:
        summary["accuracy_drop"] = summary["accuracy_orig"] - summary["accuracy_adv"]
    return summary


def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_by_model_attack(df)
    numeric_cols = summary.select_dtypes(include=[np.number]).columns.tolist()
    return summary.groupby("model")[numeric_cols].mean().reset_index()


def summarize_by_attack(df: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_by_model_attack(df)
    numeric_cols = summary.select_dtypes(include=[np.number]).columns.tolist()
    return summary.groupby("attack")[numeric_cols].mean().reset_index()


def summarize_by_model_attack_label(df: pd.DataFrame) -> pd.DataFrame:
    if "label_str" not in df.columns:
        return pd.DataFrame()

    agg_dict = {
        "n_samples": ("experiment", "size"),
        "pred_preserved_rate": ("pred_preserved_calc", "mean"),
        "aasr": ("aasr_strict", "mean"),
        "aasr_quality": ("aasr_quality", "mean"),
        "afs_simple_mean": ("afs_simple", "mean"),
        "afs_quality_mean": ("afs_quality", "mean"),
        "afs_stable_mean": ("afs_stable", "mean"),
        "afs_stable_median": ("afs_stable", "median"),
        "attribution_change_median": ("attribution_change", "median"),
        "cos_sim_median": ("cos_sim", "median"),
        "top10_overlap_median": ("top10_overlap", "median"),
        "margin_adv_median": ("margin_adv", "median"),
        "near_boundary_adv_010": ("near_boundary_adv_010", "mean"),
        "quality_score_median": ("quality_score", "median"),
    }
    return (
        df.groupby(["model", "attack", "experiment", "label_str"])
        .agg(**agg_dict)
        .reset_index()
    )


def make_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    ranking = summary.sort_values("afs_stable_mean", ascending=False).copy()
    ranking.insert(0, "rank_afs_stable", range(1, len(ranking) + 1))
    return ranking
