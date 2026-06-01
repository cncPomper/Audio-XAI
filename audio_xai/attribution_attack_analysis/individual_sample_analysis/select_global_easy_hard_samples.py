#!/usr/bin/env python3
"""
select_global_easy_hard_samples.py

Standalone script #1.

Cel:
- bierze zbiorczy CSV z wynikami eksperymentów, np. 01_all_results_9_experiments.csv,
- upewnia się, że istnieją kolumny AFS / AFS stable,
- agreguje wyniki GLOBALNIE po próbce, domyślnie po kolumnie `stem`,
- wybiera:
    1. top K easy samples: najwyższe mean AFS stable,
    2. top K hard samples: przypadki trudne dlatego, że mapa atrybucji nie chce się zmienić,
       czyli predykcja jest raczej zachowana, jakość nie jest problemem, ale attribution_change jest niskie,
- printuje tabele,
- zapisuje pełne CSV oraz osobne pliki .txt z samymi nazwami próbek.

Przykład:
    python select_global_easy_hard_samples.py \
        --input-csv outputs/runs/all_models__all_attacks/csv/01_all_results_9_experiments.csv \
        --output-dir case_selection \
        --top-k 10

Jeśli CSV nie ma kolumn afs_stable / attribution_change / quality_score,
skrypt spróbuje je policzyć z kolumn:
pred_orig, pred_adv, prob_orig, prob_adv, cos_sim, top10_overlap oraz metryk jakości.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_QUALITY_THRESHOLDS = {
    "pesq": 3.0,
    "stoi": 0.90,
    "visqol": 3.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select global top easy/hard samples using AFS stable."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        type=Path,
        help="Path to combined results CSV, e.g. 01_all_results_9_experiments.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("case_selection"),
        type=Path,
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--sample-col",
        default="stem",
        help="Column used as sample identifier. Usually `stem`.",
    )
    parser.add_argument(
        "--top-k",
        default=10,
        type=int,
        help="Number of easy and hard samples to select.",
    )
    parser.add_argument(
        "--decision-threshold",
        default=0.5,
        type=float,
        help="Decision threshold for fake score.",
    )
    parser.add_argument(
        "--min-hard-pred-preserved",
        default=0.90,
        type=float,
        help=(
            "Hard attr-stable samples must have at least this global prediction-preserved rate. "
            "This avoids selecting cases that are hard only because prediction flips."
        ),
    )
    parser.add_argument(
        "--min-hard-quality",
        default=0.60,
        type=float,
        help=(
            "Hard attr-stable samples must have at least this mean quality_score. "
            "This avoids selecting cases that are hard mainly because audio quality fails."
        ),
    )
    parser.add_argument(
        "--hard-sort",
        choices=["lowest_afs", "lowest_attr_change"],
        default="lowest_afs",
        help=(
            "How to rank hard samples after filtering. "
            "`lowest_afs` prefers globally lowest AFS stable. "
            "`lowest_attr_change` focuses directly on the smallest attribution change."
        ),
    )
    parser.add_argument(
        "--print-all-cols",
        action="store_true",
        help="Print more columns in console output.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: Iterable[str], context: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        suffix = f" for {context}" if context else ""
        raise ValueError(f"Missing required columns{suffix}: {missing}")


def normalize_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    quality_cols = []

    if "pesq" in df.columns:
        df["pesq_norm"] = ((df["pesq"] - 1.0) / (4.5 - 1.0)).clip(0, 1)
        quality_cols.append("pesq_norm")

    if "stoi" in df.columns:
        df["stoi_norm"] = df["stoi"].clip(0, 1)
        quality_cols.append("stoi_norm")

    if "visqol" in df.columns:
        df["visqol_norm"] = ((df["visqol"] - 1.0) / (5.0 - 1.0)).clip(0, 1)
        quality_cols.append("visqol_norm")

    if "peaq" in df.columns:
        df["peaq_norm"] = ((df["peaq"] + 4.0) / 4.0).clip(0, 1)
        quality_cols.append("peaq_norm")

    if "zimtohrli" in df.columns:
        df["zimtohrli_norm"] = (df["zimtohrli"] / 5.0).clip(0, 1)
        quality_cols.append("zimtohrli_norm")

    if quality_cols:
        df["quality_score"] = df[quality_cols].mean(axis=1)
    else:
        print("WARNING: No audio quality metrics found. Setting quality_score = 1.0.")
        df["quality_score"] = 1.0

    return df


def add_missing_case_metrics(df: pd.DataFrame, decision_threshold: float) -> pd.DataFrame:
    """
    Adds pred_preserved_calc, attribution_change, quality_score, margin_safety,
    afs_simple, afs_quality, afs_stable if missing.
    """
    df = df.copy()

    require_columns(
        df,
        ["pred_orig", "pred_adv", "prob_orig", "prob_adv", "cos_sim", "top10_overlap"],
        context="AFS computation",
    )

    if "pred_preserved_calc" not in df.columns:
        df["pred_preserved_calc"] = (df["pred_orig"] == df["pred_adv"]).astype(int)

    if "margin_adv" not in df.columns:
        df["margin_adv"] = (df["prob_adv"] - decision_threshold).abs()

    if "margin_orig" not in df.columns:
        df["margin_orig"] = (df["prob_orig"] - decision_threshold).abs()

    if "margin_drop" not in df.columns:
        df["margin_drop"] = df["margin_orig"] - df["margin_adv"]

    if "margin_safety" not in df.columns:
        # For threshold=0.5, max useful distance from threshold is 0.5.
        denom = max(decision_threshold, 1e-12)
        df["margin_safety"] = (df["margin_adv"] / denom).clip(0, 1)

    if "abs_score_shift" not in df.columns:
        df["score_shift"] = df["prob_adv"] - df["prob_orig"]
        df["abs_score_shift"] = df["score_shift"].abs()

    if "attribution_change" not in df.columns:
        if df["cos_sim"].min(skipna=True) < 0:
            cos_norm = ((df["cos_sim"] + 1.0) / 2.0).clip(0, 1)
        else:
            cos_norm = df["cos_sim"].clip(0, 1)

        top10_norm = df["top10_overlap"].clip(0, 1)

        df["cos_change"] = 1.0 - cos_norm
        df["top10_change"] = 1.0 - top10_norm
        df["attribution_change"] = 0.5 * (df["cos_change"] + df["top10_change"])

    if "quality_score" not in df.columns:
        df = normalize_quality_scores(df)

    if "afs_simple" not in df.columns:
        df["afs_simple"] = df["pred_preserved_calc"] * df["attribution_change"]

    if "afs_quality" not in df.columns:
        df["afs_quality"] = df["afs_simple"] * df["quality_score"]

    if "afs_stable" not in df.columns:
        df["afs_stable"] = df["afs_quality"] * df["margin_safety"]

    return df


def make_sample_summary(df: pd.DataFrame, sample_col: str) -> pd.DataFrame:
    optional_cols = {
        "label_str": ("label_str", lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
    }

    agg_spec = {
        "n_cases": (sample_col, "size"),
        "n_models": ("model", "nunique") if "model" in df.columns else (sample_col, "size"),
        "n_attacks": ("attack", "nunique") if "attack" in df.columns else (sample_col, "size"),

        "afs_stable_mean": ("afs_stable", "mean"),
        "afs_stable_median": ("afs_stable", "median"),
        "afs_stable_max": ("afs_stable", "max"),
        "afs_stable_min": ("afs_stable", "min"),

        "afs_quality_mean": ("afs_quality", "mean"),
        "afs_simple_mean": ("afs_simple", "mean"),

        "pred_preserved_rate": ("pred_preserved_calc", "mean"),
        "attribution_change_mean": ("attribution_change", "mean"),
        "attribution_change_median": ("attribution_change", "median"),
        "cos_sim_median": ("cos_sim", "median"),
        "top10_overlap_median": ("top10_overlap", "median"),

        "quality_score_mean": ("quality_score", "mean"),
        "quality_score_median": ("quality_score", "median"),
        "margin_adv_mean": ("margin_adv", "mean"),
        "margin_adv_median": ("margin_adv", "median"),
        "margin_drop_mean": ("margin_drop", "mean"),
        "abs_score_shift_mean": ("abs_score_shift", "mean"),
        "abs_score_shift_median": ("abs_score_shift", "median"),
    }

    for out_col, spec in optional_cols.items():
        if spec[0] in df.columns:
            agg_spec[out_col] = spec

    for metric in ["pesq", "stoi", "visqol", "peaq", "zimtohrli"]:
        if metric in df.columns:
            agg_spec[f"{metric}_median"] = (metric, "median")

    summary = df.groupby(sample_col).agg(**agg_spec).reset_index()

    # Best/worst concrete case for each sample.
    idx_best = df.groupby(sample_col)["afs_stable"].idxmax()
    best = df.loc[idx_best, [sample_col, "afs_stable"]].rename(
        columns={"afs_stable": "best_case_afs_stable"}
    )

    if "model" in df.columns:
        best["best_model"] = df.loc[idx_best, "model"].values
    if "attack" in df.columns:
        best["best_attack"] = df.loc[idx_best, "attack"].values

    idx_worst = df.groupby(sample_col)["afs_stable"].idxmin()
    worst = df.loc[idx_worst, [sample_col, "afs_stable"]].rename(
        columns={"afs_stable": "worst_case_afs_stable"}
    )

    if "model" in df.columns:
        worst["worst_model"] = df.loc[idx_worst, "model"].values
    if "attack" in df.columns:
        worst["worst_attack"] = df.loc[idx_worst, "attack"].values

    summary = summary.merge(best, on=sample_col, how="left")
    summary = summary.merge(worst, on=sample_col, how="left")

    return summary


def add_reason_tags(row: pd.Series, kind: str) -> str:
    tags = []

    if row.get("pred_preserved_rate", 0) >= 0.95:
        tags.append("prediction_preserved")
    elif row.get("pred_preserved_rate", 0) < 0.75:
        tags.append("prediction_often_flips")

    if row.get("attribution_change_mean", 0) >= 0.80:
        tags.append("strong_attr_change")
    elif row.get("attribution_change_mean", 0) <= 0.35:
        tags.append("attribution_stable")

    if row.get("cos_sim_median", 1) <= 0.20:
        tags.append("low_cos_sim")
    elif row.get("cos_sim_median", 0) >= 0.70:
        tags.append("high_cos_sim")

    if row.get("top10_overlap_median", 1) <= 0.10:
        tags.append("low_top10_overlap")
    elif row.get("top10_overlap_median", 0) >= 0.50:
        tags.append("high_top10_overlap")

    if row.get("quality_score_mean", 0) >= 0.80:
        tags.append("high_quality")
    elif row.get("quality_score_mean", 1) < 0.60:
        tags.append("low_quality")

    if row.get("margin_adv_mean", 0) >= 0.25:
        tags.append("safe_margin")
    elif row.get("margin_adv_mean", 1) < 0.10:
        tags.append("near_decision_boundary")

    if kind == "easy":
        tags.insert(0, "easy_by_high_afs_stable")
    else:
        tags.insert(0, "hard_by_attr_stability")

    return ";".join(tags)


def save_sample_names(df: pd.DataFrame, sample_col: str, path: Path) -> None:
    names = df[sample_col].astype(str).drop_duplicates().tolist()
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def print_table(title: str, df: pd.DataFrame, cols: list[str]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    existing = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(df[existing].to_string(index=False))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    if args.sample_col not in df.columns:
        raise ValueError(
            f"Sample column `{args.sample_col}` not found. Available columns: {list(df.columns)}"
        )

    df = add_missing_case_metrics(df, decision_threshold=args.decision_threshold)

    # Save enriched row-level data for traceability.
    df.to_csv(args.output_dir / "01_all_cases_with_case_metrics.csv", index=False)

    sample_summary = make_sample_summary(df, args.sample_col)

    # Easy: globally high mean AFS stable.
    easy = (
        sample_summary.sort_values(
            ["afs_stable_mean", "afs_stable_median", "afs_stable_max"],
            ascending=[False, False, False],
        )
        .head(args.top_k)
        .copy()
    )
    easy["case_group"] = "easy"
    easy["reason_tags"] = easy.apply(lambda row: add_reason_tags(row, "easy"), axis=1)

    # Hard attr-stable: not due to prediction flips or quality; map just remains similar.
    hard_pool = sample_summary[
        (sample_summary["pred_preserved_rate"] >= args.min_hard_pred_preserved)
        & (sample_summary["quality_score_mean"] >= args.min_hard_quality)
    ].copy()

    if hard_pool.empty:
        print(
            "WARNING: Hard pool is empty after filters. "
            "Relax --min-hard-pred-preserved or --min-hard-quality."
        )
        hard_pool = sample_summary.copy()

    if args.hard_sort == "lowest_attr_change":
        hard = (
            hard_pool.sort_values(
                ["attribution_change_mean", "afs_stable_mean", "cos_sim_median", "top10_overlap_median"],
                ascending=[True, True, False, False],
            )
            .head(args.top_k)
            .copy()
        )
    else:
        hard = (
            hard_pool.sort_values(
                ["afs_stable_mean", "attribution_change_mean", "cos_sim_median", "top10_overlap_median"],
                ascending=[True, True, False, False],
            )
            .head(args.top_k)
            .copy()
        )

    hard["case_group"] = "hard_attr_stable"
    hard["reason_tags"] = hard.apply(lambda row: add_reason_tags(row, "hard"), axis=1)

    selected_names = pd.concat(
        [easy[[args.sample_col, "case_group"]], hard[[args.sample_col, "case_group"]]],
        ignore_index=True,
    )

    selected_details = df[df[args.sample_col].astype(str).isin(selected_names[args.sample_col].astype(str))].copy()

    # Save outputs.
    sample_summary.to_csv(args.output_dir / "02_global_sample_attackability_summary.csv", index=False)
    easy.to_csv(args.output_dir / "03_top_easy_samples_global.csv", index=False)
    hard.to_csv(args.output_dir / "04_top_hard_attr_stable_samples_global.csv", index=False)
    selected_details.to_csv(args.output_dir / "05_selected_samples_all_experiment_rows.csv", index=False)
    selected_names.to_csv(args.output_dir / "06_selected_sample_names_with_group.csv", index=False)

    save_sample_names(easy, args.sample_col, args.output_dir / "easy_sample_names.txt")
    save_sample_names(hard, args.sample_col, args.output_dir / "hard_sample_names.txt")
    save_sample_names(selected_names, args.sample_col, args.output_dir / "selected_sample_names_all.txt")

    # Print concise tables.
    base_cols = [
        args.sample_col,
        "label_str",
        "n_cases",
        "n_models",
        "n_attacks",
        "afs_stable_mean",
        "afs_stable_median",
        "afs_stable_max",
        "pred_preserved_rate",
        "attribution_change_mean",
        "cos_sim_median",
        "top10_overlap_median",
        "quality_score_mean",
        "margin_adv_mean",
        "abs_score_shift_mean",
        "best_model",
        "best_attack",
        "best_case_afs_stable",
        "reason_tags",
    ]

    if args.print_all_cols:
        base_cols = list(easy.columns)

    print_table(f"TOP {args.top_k} EASY GLOBAL SAMPLES by mean AFS stable", easy, base_cols)
    print_table(f"TOP {args.top_k} HARD GLOBAL SAMPLES: attribution-stable / map does not change", hard, base_cols)

    print("\nSaved outputs to:", args.output_dir.resolve())
    print("Easy sample names:", (args.output_dir / "easy_sample_names.txt").resolve())
    print("Hard sample names:", (args.output_dir / "hard_sample_names.txt").resolve())
    print("All selected sample names:", (args.output_dir / "selected_sample_names_all.txt").resolve())


if __name__ == "__main__":
    main()
