from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Pozwala uruchamiać projekt bez instalacji jako pakiet:
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from attr_attack_analysis.config import apply_cli_overrides, load_config
from attr_attack_analysis.constants import PLOT_GROUPS
from attr_attack_analysis.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza kruchości atrybucji dla układu modele x metody ataku."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Ścieżka do pliku config.yaml."
    )
    parser.add_argument(
        "--model",
        nargs="*",
        default=None,
        help="Modele do uwzględnienia, np. --model AST VGG.",
    )
    parser.add_argument(
        "--attack",
        nargs="*",
        default=None,
        help="Ataki do uwzględnienia, np. --attack PGD FGSM.",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Nadpisuje output_dir z config.yaml."
    )

    parser.add_argument(
        "--plots",
        nargs="*",
        default=["all"],
        choices=sorted(PLOT_GROUPS),
        help="Grupy wykresów: all, heatmaps, rankings, tradeoffs, diagnostics, labels, audio, scores, distributions, thresholds, cases, interactive.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Nie generuj wykresów.")
    parser.add_argument("--no-csv", action="store_true", help="Nie zapisuj CSV.")
    parser.add_argument(
        "--only-summary",
        action="store_true",
        help="Pomiń cięższe wykresy szczegółowe, np. score scatter.",
    )
    parser.add_argument(
        "--only-plots",
        action="store_true",
        help="Odtwórz wykresy na podstawie istniejących CSV w katalogu runa.",
    )

    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="Próg decyzyjny score klasy fake.",
    )
    parser.add_argument(
        "--cos-threshold", type=float, default=None, help="Próg cos_sim dla AASR."
    )
    parser.add_argument(
        "--top10-threshold",
        type=float,
        default=None,
        help="Próg top10_overlap dla AASR.",
    )
    parser.add_argument(
        "--near-boundary-margin",
        type=float,
        default=None,
        help="Margines uznawany za blisko progu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(
        config,
        output_dir=args.output_dir,
        decision_threshold=args.decision_threshold,
        cos_threshold=args.cos_threshold,
        top10_threshold=args.top10_threshold,
        near_boundary_margin=args.near_boundary_margin,
    )

    run_pipeline(
        config=config,
        selected_models=args.model,
        selected_attacks=args.attack,
        plot_groups=args.plots,
        save_csv=not args.no_csv,
        make_plots=not args.no_plots,
        only_summary=args.only_summary,
        only_plots=args.only_plots,
    )


if __name__ == "__main__":
    main()
