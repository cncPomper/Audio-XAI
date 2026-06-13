from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt(value: float | int | str, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def generate_markdown_report(
    summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    attack_summary: pd.DataFrame,
    output_path: str | Path,
    threshold_sensitivity: pd.DataFrame | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Attribution Attack Analysis Report")
    lines.append("")
    lines.append("## Definicja sukcesu")
    lines.append("")
    lines.append(
        "Udany atak nie musi zmieniać klasy predykcji. W tej analizie atak jest "
        "uznawany za skuteczny wtedy, gdy predykcja zostaje zachowana, ale mapa "
        "atrybucji ulega istotnej zmianie."
    )
    lines.append("")
    lines.append("Najważniejsze metryki:")
    lines.append("")
    lines.append(
        "- **AASR**: metryka progowa: `pred_orig == pred_adv`, `cos_sim < próg`, `top10_overlap < próg`."
    )
    lines.append(
        "- **AFS simple**: ciągła kruchość atrybucji przy zachowanej predykcji."
    )
    lines.append("- **AFS quality**: AFS simple ważony znormalizowaną jakością audio.")
    lines.append(
        "- **AFS stable**: AFS quality ważony odległością od progu decyzyjnego po ataku."
    )
    lines.append(
        "- **CI 95%**: przybliżone przedziały ufności średnich, pomocne przy porównywaniu rankingów."
    )
    lines.append("")

    if not summary.empty:
        top_combo = summary.sort_values("afs_stable_mean", ascending=False).iloc[0]
        lines.append("## Najlepsza kombinacja model x atak według AFS stable")
        lines.append("")
        lines.append(f"- Model: **{top_combo['model']}**")
        lines.append(f"- Atak: **{top_combo['attack']}**")
        lines.append(f"- AFS stable mean: **{_fmt(top_combo['afs_stable_mean'])}**")
        if "afs_stable_ci_low" in top_combo and "afs_stable_ci_high" in top_combo:
            lines.append(
                f"- AFS stable 95% CI: **{_fmt(top_combo['afs_stable_ci_low'])}-{_fmt(top_combo['afs_stable_ci_high'])}**"
            )
        lines.append(f"- AASR: **{_fmt(top_combo['aasr'])}**")
        lines.append(
            f"- Pred preserved rate: **{_fmt(top_combo['pred_preserved_rate'])}**"
        )
        lines.append(
            f"- Quality score median: **{_fmt(top_combo['quality_score_median'])}**"
        )
        lines.append("")

    if not model_summary.empty:
        best_model = model_summary.sort_values("afs_stable_mean", ascending=False).iloc[
            0
        ]
        lines.append("## Najbardziej podatny model średnio po atakach")
        lines.append("")
        lines.append(f"- Model: **{best_model['model']}**")
        lines.append(f"- AFS stable mean: **{_fmt(best_model['afs_stable_mean'])}**")
        lines.append(f"- AASR: **{_fmt(best_model['aasr'])}**")
        lines.append("")

    if not attack_summary.empty:
        best_attack = attack_summary.sort_values(
            "afs_stable_mean", ascending=False
        ).iloc[0]
        lines.append("## Najskuteczniejsza metoda ataku średnio po modelach")
        lines.append("")
        lines.append(f"- Atak: **{best_attack['attack']}**")
        lines.append(f"- AFS stable mean: **{_fmt(best_attack['afs_stable_mean'])}**")
        lines.append(f"- AASR: **{_fmt(best_attack['aasr'])}**")
        lines.append("")

    if threshold_sensitivity is not None and not threshold_sensitivity.empty:
        lines.append("## Czułość AASR na progi")
        lines.append("")
        lines.append(
            "Wygenerowano `csv/11_threshold_sensitivity.csv` oraz wykresy w `figures/thresholds/`. "
            "Te pliki pokazują, czy ranking ataków jest stabilny, czy zależy od arbitralnego wyboru "
            "`cos_threshold` i `top10_threshold`."
        )
        lines.append("")

    lines.append("## Najważniejsze pliki wynikowe")
    lines.append("")
    lines.append("- `csv/01_all_results.csv` — wszystkie próbki z dodanymi metrykami.")
    lines.append(
        "- `csv/02_summary_by_model_attack.csv` — główne porównanie par model x atak z percentylami i CI."
    )
    lines.append("- `csv/06_ranking_by_afs_stable.csv` — ranking kombinacji.")
    lines.append(
        "- `csv/07_top_success_cases.csv` — próbki najbardziej reprezentatywne dla udanego ataku."
    )
    lines.append(
        "- `csv/08_top_attribution_change_cases.csv` — próbki z największą zmianą wyjaśnienia."
    )
    lines.append(
        "- `csv/09_boundary_cases.csv` — próbki blisko progu decyzyjnego po ataku."
    )
    lines.append(
        "- `csv/11_threshold_sensitivity.csv` — AASR liczony na siatce progów."
    )
    lines.append("- `figures/heatmaps/` — porównywalne heatmapy ze stałą skalą.")
    lines.append("- `figures/distributions/` — boxploty, ECDF i rozkłady próbek.")
    lines.append("- `figures/thresholds/` — analiza czułości progów AASR.")
    lines.append(
        "- `interactive/dashboard.html` — interaktywny dashboard Plotly, jeżeli `plotly` jest zainstalowany."
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
