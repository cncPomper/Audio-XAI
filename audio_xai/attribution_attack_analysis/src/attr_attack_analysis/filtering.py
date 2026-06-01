from __future__ import annotations

from .config import ExperimentSpec


def filter_experiments(
    experiments: list[ExperimentSpec],
    selected_models: list[str] | None = None,
    selected_attacks: list[str] | None = None,
) -> list[ExperimentSpec]:
    selected_models_set = set(selected_models or [])
    selected_attacks_set = set(selected_attacks or [])

    filtered = []
    for exp in experiments:
        if selected_models_set and exp.model not in selected_models_set:
            continue
        if selected_attacks_set and exp.attack not in selected_attacks_set:
            continue
        filtered.append(exp)

    if not filtered:
        raise ValueError(
            "Po zastosowaniu filtrów nie został żaden eksperyment. "
            f"selected_models={selected_models}, selected_attacks={selected_attacks}"
        )
    return filtered


def make_run_name(
    experiments: list[ExperimentSpec],
    selected_models: list[str] | None = None,
    selected_attacks: list[str] | None = None,
) -> str:
    all_models = sorted({e.model for e in experiments})
    all_attacks = sorted({e.attack for e in experiments})

    models_part = "all_models" if not selected_models else "_".join(selected_models)
    attacks_part = "all_attacks" if not selected_attacks else "_".join(selected_attacks)

    def safe(text: str) -> str:
        return (
            text.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace("|", "_")
            .replace(":", "_")
        )

    name = f"{safe(models_part)}__{safe(attacks_part)}"

    # If user did not pass explicit filters but config contains only a subset,
    # still use all_models/all_attacks as a stable convention.
    if selected_models and sorted(selected_models) == all_models:
        name = name.replace(safe(models_part), "all_models", 1)
    if selected_attacks and sorted(selected_attacks) == all_attacks:
        name = name.replace(safe(attacks_part), "all_attacks", 1)
    return name
