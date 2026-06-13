# Attribution Attack Analysis Report

## Definicja sukcesu

Udany atak nie musi zmieniać klasy predykcji. W tej analizie atak jest uznawany za skuteczny wtedy, gdy predykcja zostaje zachowana, ale mapa atrybucji ulega istotnej zmianie.

Najważniejsze metryki:

- **AASR**: metryka progowa: `pred_orig == pred_adv`, `cos_sim < próg`, `top10_overlap < próg`.
- **AFS simple**: ciągła kruchość atrybucji przy zachowanej predykcji.
- **AFS quality**: AFS simple ważony znormalizowaną jakością audio.
- **AFS stable**: AFS quality ważony odległością od progu decyzyjnego po ataku.
- **CI 95%**: przybliżone przedziały ufności średnich, pomocne przy porównywaniu rankingów.

## Najlepsza kombinacja model x atak według AFS stable

- Model: **VGG**
- Atak: **X-Shift**
- AFS stable mean: **0.6464**
- AFS stable 95% CI: **0.5989-0.6939**
- AASR: **0.9200**
- Pred preserved rate: **0.9200**
- Quality score median: **0.7468**

## Najbardziej podatny model średnio po atakach

- Model: **AST**
- AFS stable mean: **0.5633**
- AASR: **0.7933**

## Najskuteczniejsza metoda ataku średnio po modelach

- Atak: **Psychoacoustic**
- AFS stable mean: **0.5101**
- AASR: **0.3967**

## Czułość AASR na progi

Wygenerowano `csv/11_threshold_sensitivity.csv` oraz wykresy w `figures/thresholds/`. Te pliki pokazują, czy ranking ataków jest stabilny, czy zależy od arbitralnego wyboru `cos_threshold` i `top10_threshold`.

## Najważniejsze pliki wynikowe

- `csv/01_all_results.csv` — wszystkie próbki z dodanymi metrykami.
- `csv/02_summary_by_model_attack.csv` — główne porównanie par model x atak z percentylami i CI.
- `csv/06_ranking_by_afs_stable.csv` — ranking kombinacji.
- `csv/07_top_success_cases.csv` — próbki najbardziej reprezentatywne dla udanego ataku.
- `csv/08_top_attribution_change_cases.csv` — próbki z największą zmianą wyjaśnienia.
- `csv/09_boundary_cases.csv` — próbki blisko progu decyzyjnego po ataku.
- `csv/11_threshold_sensitivity.csv` — AASR liczony na siatce progów.
- `figures/heatmaps/` — porównywalne heatmapy ze stałą skalą.
- `figures/distributions/` — boxploty, ECDF i rozkłady próbek.
- `figures/thresholds/` — analiza czułości progów AASR.
- `interactive/dashboard.html` — interaktywny dashboard Plotly, jeżeli `plotly` jest zainstalowany.