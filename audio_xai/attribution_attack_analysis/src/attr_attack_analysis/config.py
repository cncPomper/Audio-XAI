from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentSpec:
    model: str
    attack: str
    path: Path

    @property
    def experiment(self) -> str:
        return f"{self.model} | {self.attack}"


@dataclass(frozen=True)
class AnalysisConfig:
    experiments: list[ExperimentSpec]
    output_dir: Path
    decision_threshold: float
    cos_threshold: float
    top10_threshold: float
    near_boundary_margin: float
    quality_thresholds: dict[str, float]


def load_config(path: str | Path) -> AnalysisConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_dir = path.parent
    experiments = []
    for item in raw.get("experiments", []):
        exp_path = Path(item["path"])
        if not exp_path.is_absolute():
            exp_path = (base_dir / exp_path).resolve()
        experiments.append(
            ExperimentSpec(
                model=str(item["model"]),
                attack=str(item["attack"]),
                path=exp_path,
            )
        )

    output_dir = Path(raw.get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()

    thresholds: dict[str, Any] = raw.get("thresholds", {}) or {}
    quality_thresholds: dict[str, float] = {
        str(k): float(v) for k, v in (raw.get("quality_thresholds", {}) or {}).items()
    }

    config = AnalysisConfig(
        experiments=experiments,
        output_dir=output_dir,
        decision_threshold=float(thresholds.get("decision_threshold", 0.5)),
        cos_threshold=float(thresholds.get("cos_threshold", 0.2)),
        top10_threshold=float(thresholds.get("top10_threshold", 0.1)),
        near_boundary_margin=float(thresholds.get("near_boundary_margin", 0.1)),
        quality_thresholds=quality_thresholds or {"pesq": 3.0, "stoi": 0.90, "visqol": 3.5},
    )
    validate_config(config)
    return config


def validate_config(config: AnalysisConfig) -> None:
    if not config.experiments:
        raise ValueError("Brak eksperymentów w config.yaml.")

    seen = set()
    duplicates = []
    for exp in config.experiments:
        key = (exp.model, exp.attack)
        if key in seen:
            duplicates.append(key)
        seen.add(key)

    if duplicates:
        raise ValueError(f"Zduplikowane pary model/attack w konfiguracji: {duplicates}")


def apply_cli_overrides(
    config: AnalysisConfig,
    output_dir: str | Path | None = None,
    decision_threshold: float | None = None,
    cos_threshold: float | None = None,
    top10_threshold: float | None = None,
    near_boundary_margin: float | None = None,
) -> AnalysisConfig:
    return AnalysisConfig(
        experiments=config.experiments,
        output_dir=Path(output_dir).resolve() if output_dir else config.output_dir,
        decision_threshold=config.decision_threshold if decision_threshold is None else decision_threshold,
        cos_threshold=config.cos_threshold if cos_threshold is None else cos_threshold,
        top10_threshold=config.top10_threshold if top10_threshold is None else top10_threshold,
        near_boundary_margin=(
            config.near_boundary_margin if near_boundary_margin is None else near_boundary_margin
        ),
        quality_thresholds=config.quality_thresholds,
    )
