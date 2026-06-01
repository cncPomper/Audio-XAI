from __future__ import annotations

PLOT_GROUPS = {
    "heatmaps",
    "rankings",
    "tradeoffs",
    "diagnostics",
    "labels",
    "audio",
    "scores",
    "distributions",
    "thresholds",
    "cases",
    "interactive",
    "all",
}

SUMMARY_COLUMNS_FOR_AGGREGATES = [
    "pred_preserved_rate",
    "aasr",
    "afs_quality_mean",
    "afs_stable_mean",
    "quality_score_median",
]

METRIC_INFO = {
    "pesq": {
        "label": "PESQ",
        "range": "-0.5-4.5",
        "direction": "wyżej = lepiej",
    },
    "stoi": {
        "label": "STOI",
        "range": "0-1",
        "direction": "wyżej = lepiej",
    },
    "visqol": {
        "label": "ViSQOL",
        "range": "1-5",
        "direction": "wyżej = lepiej",
    },
    "peaq": {
        "label": "PEAQ",
        "range": "-4-0",
        "direction": "bliżej 0 = lepiej",
    },
    "zimtohrli": {
        "label": "Zimtohrli",
        "range": "0-5",
        "direction": "wyżej = lepiej",
    },
    "cdpam": {
        "label": "CDPAM",
        "range": "0-1",
        "direction": "wyżej = lepiej",
    },
    "cos_sim": {
        "label": "Cosine similarity",
        "range": "0-1",
        "direction": "niżej = większa zmiana atrybucji",
    },
    "top10_overlap": {
        "label": "Top-10 overlap",
        "range": "0-1",
        "direction": "niżej = większa zmiana top regionów",
    },
}
