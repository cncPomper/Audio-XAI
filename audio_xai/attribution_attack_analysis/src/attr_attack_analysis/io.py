from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ExperimentSpec


def load_json_folder(json_dir: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    json_dir = Path(json_dir)

    if not json_dir.exists():
        raise FileNotFoundError(f"Folder nie istnieje: {json_dir}")
    if not json_dir.is_dir():
        raise NotADirectoryError(f"To nie jest folder: {json_dir}")

    for path in sorted(json_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Plik JSON nie zawiera obiektu/dict na najwyższym poziomie.")
            data["_file"] = path.name
            rows.append(data)
        except Exception as e:
            print(f"Nie udało się wczytać {path}: {e}")

    if not rows:
        raise ValueError(f"Nie znaleziono poprawnych plików JSON w: {json_dir}")

    return pd.DataFrame(rows)


def load_experiment(exp: ExperimentSpec) -> pd.DataFrame:
    df = load_json_folder(exp.path)
    df["model"] = exp.model
    df["attack"] = exp.attack
    df["experiment"] = exp.experiment
    df["_folder"] = str(exp.path)
    return df


def load_all_experiments(experiments: list[ExperimentSpec]) -> pd.DataFrame:
    frames = [load_experiment(exp) for exp in experiments]
    return pd.concat(frames, ignore_index=True)


def ensure_output_dirs(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    dirs = {
        "run": run_dir,
        "csv": run_dir / "csv",
        "figures": run_dir / "figures",
        "heatmaps": run_dir / "figures" / "heatmaps",
        "rankings": run_dir / "figures" / "rankings",
        "tradeoffs": run_dir / "figures" / "tradeoffs",
        "diagnostics": run_dir / "figures" / "diagnostics",
        "labels": run_dir / "figures" / "labels",
        "audio": run_dir / "figures" / "audio",
        "scores": run_dir / "figures" / "scores",
        "distributions": run_dir / "figures" / "distributions",
        "thresholds": run_dir / "figures" / "thresholds",
        "cases": run_dir / "figures" / "cases",
        "interactive": run_dir / "interactive",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_csv_outputs(csv_dir: str | Path, **frames: pd.DataFrame) -> None:
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        if df is None or df.empty:
            continue
        df.to_csv(csv_dir / f"{name}.csv", index=False)
