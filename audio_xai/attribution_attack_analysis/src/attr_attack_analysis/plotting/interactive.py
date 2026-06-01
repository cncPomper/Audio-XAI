from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_all_interactive_plots(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame | None,
    out_dir: str | Path,
    **kwargs,
) -> None:
    """Eksportuje lekki dashboard HTML w Plotly, jeśli biblioteka jest zainstalowana.

    Gdy Plotly nie jest dostępne, zapisuje plik README z informacją, jak włączyć dashboard.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import plotly.express as px
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except Exception:
        (out_dir / "README_plotly_missing.md").write_text(
            "# Dashboard interaktywny nie został wygenerowany\n\n"
            "Zainstaluj zależności: `pip install plotly` i uruchom analizę ponownie.\n",
            encoding="utf-8",
        )
        return

    html_parts: list[str] = [
        "<html><head><meta charset='utf-8'><title>XAI Attack Dashboard</title></head><body>"
    ]
    html_parts.append("<h1>Attribution Attack Analysis — dashboard interaktywny</h1>")
    html_parts.append(
        "<p>Wykresy pozwalają eksplorować pary model x atak oraz pojedyncze próbki przez hover.</p>"
    )

    if summary is not None and not summary.empty:
        fig = px.imshow(
            summary.pivot(index="model", columns="attack", values="afs_stable_mean"),
            zmin=0,
            zmax=1,
            color_continuous_scale="Viridis",
            text_auto=".3f",
            title="AFS stable — heatmapa interaktywna",
        )
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

        if {
            "pred_preserved_rate",
            "attribution_change_mean",
            "quality_score_median",
            "aasr",
        }.issubset(summary.columns):
            fig = px.scatter(
                summary,
                x="pred_preserved_rate",
                y="attribution_change_mean",
                size="quality_score_median",
                color="aasr",
                hover_data=["model", "attack", "afs_stable_mean", "n_samples"],
                title="Trade-off: zachowanie predykcji vs zmiana atrybucji",
                range_x=[0, 1],
            )
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    if (
        df is not None
        and not df.empty
        and {"prob_orig", "prob_adv", "afs_stable", "quality_score"}.issubset(
            df.columns
        )
    ):
        hover = [
            c
            for c in [
                "model",
                "attack",
                "_file",
                "label_str",
                "cos_sim",
                "top10_overlap",
                "attribution_change",
                "margin_adv",
            ]
            if c in df.columns
        ]
        fig = px.scatter(
            df,
            x="prob_orig",
            y="prob_adv",
            color="afs_stable",
            size="quality_score",
            facet_col="attack" if df["attack"].nunique() <= 4 else None,
            hover_data=hover,
            title="Próbki: score przed i po ataku, kolor = AFS stable, rozmiar = jakość",
            range_x=[0, 1],
            range_y=[0, 1],
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    if (
        df is not None
        and not df.empty
        and {"cos_sim", "top10_overlap", "afs_stable"}.issubset(df.columns)
    ):
        hover = [
            c
            for c in [
                "model",
                "attack",
                "_file",
                "label_str",
                "quality_score",
                "margin_adv",
            ]
            if c in df.columns
        ]
        fig = px.scatter(
            df,
            x="cos_sim",
            y="top10_overlap",
            color="afs_stable",
            hover_data=hover,
            title="Próbki: podobieństwo kosinusowe vs overlap top-k",
        )
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    if threshold_sensitivity is not None and not threshold_sensitivity.empty:
        compact = threshold_sensitivity.groupby(
            ["cos_threshold", "top10_threshold"], as_index=False
        )["aasr"].mean()
        fig = px.imshow(
            compact.pivot(
                index="cos_threshold", columns="top10_threshold", values="aasr"
            ),
            zmin=0,
            zmax=1,
            color_continuous_scale="Viridis",
            title="Średnia czułość AASR na progi, agregacja po parach model x atak",
        )
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    html_parts.append("</body></html>")
    (out_dir / "dashboard.html").write_text("\n".join(html_parts), encoding="utf-8")
