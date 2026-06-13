# Attribution Attack Analysis

A project for comparing experiments of the type:

```text
ML models x attack methods
```

The default layout assumes **3 models x 3 attack methods = 9 folders with JSON files**, but the pipeline also works for any subset, e.g. a single model, a single attack, or a single model-attack pair.

The project was originally a single-file script `compare_9_model_attack_experiments.py`, but has been split into modules to make it easy to:

- filter experiments with CLI flags,
- enable and disable plot groups,
- change AASR thresholds without editing code,
- add new metrics,
- run analysis for selected models or attacks only,
- generate only CSVs or only selected plots.

---

## 1. Goal of the Analysis

In this project **a successful attack does not mean a change in prediction class**.

The goal is:

```text
preserve the model's prediction
+
change / invert the attribution map
+
preserve audio quality
```

So an ideal attack looks like this:

```text
pred_orig == pred_adv
cos_sim low
top10_overlap low
PESQ/STOI/ViSQOL good
model score not close to the decision boundary
```

---

## 2. Required JSON Format

Each experiment folder should contain multiple `.json` files, where each file corresponds to one sample.

Minimum required columns:

```json
{
  "pred_orig": 1,
  "prob_orig": 0.999764,
  "pred_adv": 1,
  "prob_adv": 0.999763,
  "cos_sim": 0.049832,
  "top10_overlap": 0.005853
}
```

Recommended additional fields:

```json
{
  "index": 0,
  "stem": "fake_02868_suno_1",
  "label": 1,
  "label_str": "fake",
  "correct_orig": 1,
  "correct_adv": 1,
  "pred_preserved": 1,
  "delta_linf": 0.01,
  "pesq": 3.756158,
  "stoi": 0.989188,
  "visqol": 4.073729,
  "peaq": -1.961859,
  "zimtohrli": 3.57782,
  "ok": true
}
```

### Meaning of `prob_orig` and `prob_adv`

In this project `prob_orig` and `prob_adv` are treated as a **continuous score for the `fake` class**, not as a probability of the true class.

At threshold `0.5`:

```text
score < 0.5  → pred = 0 → real
score >= 0.5 → pred = 1 → fake
```

Therefore `real` samples naturally have scores close to 0, and `fake` samples have scores close to 1.

---

## 3. Installation

In the project directory:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install -r requirements.txt
```

Requirements:

```text
numpy
pandas
matplotlib
PyYAML
```

---

## 4. Project Structure

```text
attribution_attack_analysis/
├── README.md
├── requirements.txt
├── config.yaml
├── run_analysis.py
├── outputs/
│   ├── csv/
│   └── figures/
│       ├── heatmaps/
│       ├── rankings/
│       ├── tradeoffs/
│       ├── diagnostics/
│       ├── labels/
│       ├── audio/
│       └── scores/
└── src/
    └── attr_attack_analysis/
        ├── __init__.py
        ├── constants.py
        ├── config.py
        ├── filtering.py
        ├── io.py
        ├── metrics.py
        ├── summaries.py
        ├── report.py
        ├── pipeline.py
        └── plotting/
            ├── __init__.py
            ├── common.py
            ├── heatmaps.py
            ├── rankings.py
            ├── tradeoffs.py
            ├── diagnostics.py
            ├── labels.py
            ├── audio.py
            └── scores.py
```

---

## 5. Configuration: `config.yaml`

In `config.yaml` you define all experiments.

Example for 9 folders:

```yaml
experiments:
  - model: AST
    attack: Attack_1
    path: /path/to/AST_Attack_1
  - model: AST
    attack: Attack_2
    path: /path/to/AST_Attack_2
  - model: AST
    attack: Attack_3
    path: /path/to/AST_Attack_3

  - model: VGG
    attack: Attack_1
    path: /path/to/VGG_Attack_1
  - model: VGG
    attack: Attack_2
    path: /path/to/VGG_Attack_2
  - model: VGG
    attack: Attack_3
    path: /path/to/VGG_Attack_3

  - model: SPECTR
    attack: Attack_1
    path: /path/to/SPECTR_Attack_1
  - model: SPECTR
    attack: Attack_2
    path: /path/to/SPECTR_Attack_2
  - model: SPECTR
    attack: Attack_3
    path: /path/to/SPECTR_Attack_3

output_dir: outputs

thresholds:
  decision_threshold: 0.5
  cos_threshold: 0.2
  top10_threshold: 0.1
  near_boundary_margin: 0.1

quality_thresholds:
  pesq: 3.0
  stoi: 0.90
  visqol: 3.5
```

Paths can be absolute or relative to the location of `config.yaml`.

---

## 6. Running the Analysis

### Full analysis for all experiments

```bash
python run_analysis.py --config config.yaml --plots all
```

Results go to:

```text
outputs/runs/all_models__all_attacks/
```

### Single model only

```bash
python run_analysis.py --config config.yaml --model AST
```

Results:

```text
outputs/runs/AST__all_attacks/
```

### Single attack only

```bash
python run_analysis.py --config config.yaml --attack Attack_1
```

Results:

```text
outputs/runs/all_models__Attack_1/
```

### Single model-attack pair

```bash
python run_analysis.py --config config.yaml --model AST --attack Attack_1
```

Results:

```text
outputs/runs/AST__Attack_1/
```

### Multiple models and multiple attacks

```bash
python run_analysis.py \
  --config config.yaml \
  --model AST VGG \
  --attack Attack_1 Attack_3
```

---

## 7. CLI Flags

### Data selection

```bash
--model AST VGG
```

Restricts analysis to the specified models.

```bash
--attack Attack_1 Attack_2
```

Restricts analysis to the specified attacks.

If a flag is not provided, all values from `config.yaml` are used.

---

### Plot selection

```bash
--plots all
```

Generates all plot groups.

Available groups:

```text
heatmaps
rankings
tradeoffs
diagnostics
labels
audio
scores
all
```

Examples:

```bash
python run_analysis.py --config config.yaml --plots heatmaps rankings
```

```bash
python run_analysis.py --config config.yaml --plots diagnostics labels
```

```bash
python run_analysis.py --config config.yaml --plots scores
```

---

### Disable plots

```bash
python run_analysis.py --config config.yaml --no-plots
```

Computes metrics and saves CSVs, but does not create plots.

---

### Disable CSVs

```bash
python run_analysis.py --config config.yaml --no-csv
```

Creates plots, but does not save CSV tables.

---

### Summaries only (lighter mode)

```bash
python run_analysis.py --config config.yaml --only-summary
```

Skips heavier detailed plots, e.g. scatter `prob_orig` vs `prob_adv` for each model-attack pair.

---

### Regenerate plots from existing CSVs

```bash
python run_analysis.py --config config.yaml --only-plots
```

This option expects that CSVs already exist in the corresponding run directory, e.g.:

```text
outputs/runs/all_models__all_attacks/csv/
```

---

### Override thresholds from CLI

```bash
python run_analysis.py \
  --config config.yaml \
  --decision-threshold 0.5 \
  --cos-threshold 0.25 \
  --top10-threshold 0.15 \
  --near-boundary-margin 0.1
```

---

## 8. Metrics

### `pred_preserved_rate`

The fraction of samples for which:

```text
pred_orig == pred_adv
```

For this project's goal: higher = better, because the attack should not change the class.

---

### `cos_sim`

Cosine similarity of attribution maps before and after the attack.

Interpretation:

```text
high cos_sim → maps are similar
low cos_sim  → maps have changed
```

For an attribution attack: lower = better.

---

### `top10_overlap`

The fraction of shared elements in the top 10 most important positions of the attribution map before and after the attack.

Example:

```text
top10 before: A B C D E F G H I J
top10 after:  A B X Y Z K L M N O
shared: A, B

top10_overlap = 2/10 = 0.2
```

For an attribution attack: lower = better.

---

### `AASR`

**Attribution Attack Success Rate**.

A threshold-based metric:

```text
AASR = mean(
    pred_orig == pred_adv
    AND cos_sim < cos_threshold
    AND top10_overlap < top10_threshold
)
```

Interpretation:

```text
higher AASR = more samples formally satisfy the attribution attack success conditions
```

Diagnostic variants:

```text
aasr_cos_only
aasr_top10_only
aasr_either
aasr
```

`aasr` is the strict version, meaning both `cos_sim` and `top10_overlap` must pass the threshold simultaneously.

---

### `AFS simple`

**Attribution Fragility Score** — a continuous metric of attribution fragility:

```text
AFS_simple = pred_preserved x 0.5 x [(1 - cos_sim_norm) + (1 - top10_overlap_norm)]
```

Interpretation:

```text
higher = greater attribution change with preserved prediction
```

This is less arbitrary than AASR because it does not depend as strongly on hard thresholds.

---

### `AFS quality`

AFS incorporating audio quality:

```text
AFS_quality = AFS_simple x quality_score
```

`quality_score` is the mean of the normalised audio quality metrics.

---

### `AFS stable`

The most rigorous final metric:

```text
AFS_stable = AFS_quality x margin_safety
```

Where:

```text
margin_safety = clip(abs(prob_adv - decision_threshold) / decision_threshold, 0, 1)
```

Interpretation:

```text
high AFS_stable means:
- prediction was preserved,
- attribution map changed significantly,
- audio quality was maintained,
- post-attack score is not hovering near the decision boundary.
```

This is the primary metric for model x attack ranking.

---

### `quality_score`

Normalised audio quality in the range 0–1.

Normalisations:

```text
PESQ_norm      = clip((PESQ - 1.0) / (4.5 - 1.0), 0, 1)
STOI_norm      = clip(STOI, 0, 1)
ViSQOL_norm    = clip((ViSQOL - 1.0) / (5.0 - 1.0), 0, 1)
PEAQ_norm      = clip((PEAQ + 4.0) / 4.0, 0, 1)
Zimtohrli_norm = clip(Zimtohrli / 5.0, 0, 1)
```

If a metric does not exist, it is skipped.

---

## 9. CSV Results

For each run, results are saved in:

```text
outputs/runs/<run_name>/csv/
```

Files:

```text
01_all_results.csv
02_summary_by_model_attack.csv
03_summary_by_model.csv
04_summary_by_attack.csv
05_summary_by_model_attack_label.csv
06_ranking_by_afs_stable.csv
```

### `01_all_results.csv`

All samples from all selected experiments plus additional metric columns.

Key columns added by the pipeline:

```text
model
attack
experiment
pred_preserved_calc
score_shift
abs_score_shift
margin_orig
margin_adv
margin_drop
margin_safety
near_boundary_adv_005
near_boundary_adv_010
cos_sim_norm
top10_overlap_norm
cos_change
top10_change
attribution_change
attr_changed_cos
attr_changed_top10
attr_changed_either
attr_changed_both
aasr_cos_only
aasr_top10_only
aasr_either
aasr_strict
aasr_quality
quality_score
afs_simple
afs_quality
afs_stable
```

### `02_summary_by_model_attack.csv`

Main table for comparing all 9 combinations.

Key columns:

```text
model
attack
n_samples
pred_preserved_rate
aasr
aasr_quality
afs_simple_mean
afs_quality_mean
afs_stable_mean
attribution_change_mean
cos_sim_median
top10_overlap_median
quality_score_median
margin_adv_median
near_boundary_adv_010
```

### `03_summary_by_model.csv`

Aggregation by model, averaged across selected attacks.

Answers the question:

```text
which model has the most fragile attributions overall?
```

### `04_summary_by_attack.csv`

Aggregation by attack method, averaged across selected models.

Answers the question:

```text
which attack method is most effective overall?
```

### `05_summary_by_model_attack_label.csv`

Aggregation separately for `fake` and `real`, if the `label_str` column exists.

Answers the question:

```text
does a given method/model behave differently for fake vs real?
```

### `06_ranking_by_afs_stable.csv`

Ranking of model x attack combinations by `AFS stable`.

---

## 10. Plots

Plots are saved in:

```text
outputs/runs/<run_name>/figures/
```

### `heatmaps`

```text
01_heatmap_afs_stable_model_attack.png
02_heatmap_afs_quality_model_attack.png
03_heatmap_aasr_model_attack.png
04_heatmap_pred_preserved_model_attack.png
05_heatmap_quality_model_attack.png
06_heatmap_cos_sim_model_attack.png
07_heatmap_top10_overlap_model_attack.png
```

Most important plot:

```text
01_heatmap_afs_stable_model_attack.png
```

Shows which model x attack combination has the highest attribution fragility with preserved prediction, preserved audio quality, and stable margin from the threshold.

---

### `rankings`

```text
08_ranking_model_attack_afs_stable.png
09_ranking_model_attack_aasr.png
10_model_aggregate_comparison.png
11_attack_aggregate_comparison.png
```

Most important plot:

```text
08_ranking_model_attack_afs_stable.png
```

---

### `tradeoffs`

```text
12_tradeoff_pred_preserved_vs_attr_change_quality.png
```

Each point is one model x attack combination.

Interpretation:

```text
upper-right corner = good combination:
- prediction often preserved,
- attribution strongly changed.

larger point = better audio quality.
```

---

### `diagnostics`

```text
13_aasr_diagnostics_conditions.png
```

Shows which condition is blocking AASR:

```text
pred_preserved_rate
attr_changed_cos_rate
attr_changed_top10_rate
attr_changed_both_rate
aasr_cos_only
aasr_top10_only
aasr_either
aasr
```

This is especially important when, for example, only one model has a non-zero AASR.

---

### `labels`

```text
14_heatmap_afs_stable_label_fake.png
15_heatmap_aasr_label_fake.png
14_heatmap_afs_stable_label_real.png
15_heatmap_aasr_label_real.png
```

Shows results separately for the `fake` and `real` classes.

---

### `audio`

```text
16_audio_quality_model_attack.png
```

Compares median audio quality metrics.

---

### `scores`

```text
17_prob_orig_vs_prob_adv_<MODEL>_<ATTACK>.png
```

Scatter of the `fake` class score before and after the attack.

Interpretation at threshold 0.5:

```text
bottom-left: real → real
top-right:   fake → fake
bottom-right: fake → real
top-left:    real → fake
```

For this project's goal, points should remain in the same quadrants relative to the threshold, but the attribution map should change.

---

## 11. Automatic Markdown Report

Each run generates:

```text
outputs/runs/<run_name>/report.md
```

The report contains:

- definition of success,
- best model x attack combination by AFS stable,
- most vulnerable model averaged across attacks,
- most effective attack averaged across models,
- list of the most important result files.

---

## 12. How to Interpret Results

### High `AFS_stable`

Best case from the attribution attack perspective:

```text
prediction preserved
+
attribution strongly changed
+
audio quality maintained
+
post-attack score not close to the decision boundary
```

### High `AASR`, but low `AFS_stable`

The attack satisfies the success thresholds, but may have weaker audio quality or the score may be close to the decision boundary.

### High `pred_preserved_rate`, low `AFS`

The model preserves the prediction, but the attribution map does not change. The attribution attack is weak.

### Low `pred_preserved_rate`, high `attribution_change`

The attack changes the map, but too often also changes the prediction class. For this project, that is not an ideal success.

### Low `AASR`, but moderate `AFS`

Possible that `cos_threshold` or `top10_threshold` are too strict. Check `13_aasr_diagnostics_conditions.png`.

---

## 13. Common Use Cases

### Quick table only, no plots

```bash
python run_analysis.py --config config.yaml --no-plots
```

### Heatmaps only

```bash
python run_analysis.py --config config.yaml --plots heatmaps
```

### Rankings and trade-off only

```bash
python run_analysis.py --config config.yaml --plots rankings tradeoffs
```

### Diagnostics only — why is AASR zero?

```bash
python run_analysis.py --config config.yaml --plots diagnostics
```

### Score scatter for a specific model and attack

```bash
python run_analysis.py --config config.yaml --model AST --attack Attack_1 --plots scores
```

### Fake vs real analysis only

```bash
python run_analysis.py --config config.yaml --plots labels
```

### Relaxed AASR thresholds

```bash
python run_analysis.py \
  --config config.yaml \
  --cos-threshold 0.3 \
  --top10-threshold 0.2 \
  --plots heatmaps diagnostics
```

---

## 14. Adding a New Metric

Best place:

```text
src/attr_attack_analysis/metrics.py
```

Add a column in one of the functions:

```text
add_prediction_metrics
add_attribution_metrics
add_success_metrics
add_fragility_metrics
```

Then add aggregation in:

```text
src/attr_attack_analysis/summaries.py
```

If you want to plot it, add a plot in the appropriate module:

```text
src/attr_attack_analysis/plotting/
```

---

## 15. Adding a New Plot Group

1. Create a file, e.g.:

```text
src/attr_attack_analysis/plotting/new_group.py
```

2. Add a function:

```python
def plot_all_new_group(...):
    ...
```

3. Register it in:

```text
src/attr_attack_analysis/plotting/__init__.py
```

4. Add the group name to:

```text
src/attr_attack_analysis/constants.py
```

Then you can run:

```bash
python run_analysis.py --plots new_group
```

---

## 16. Troubleshooting

### `No valid JSON files found`

Check that the path in `config.yaml` points to a folder containing `.json` files, not the parent folder.

### `Missing required columns`

Minimum required columns are:

```text
pred_orig
pred_adv
prob_orig
prob_adv
cos_sim
top10_overlap
```

### `labels` plots are not generated

Missing column:

```text
label_str
```

### `AASR` is almost everywhere zero

Check:

```text
figures/diagnostics/13_aasr_diagnostics_conditions.png
```

Possible causes:

- `cos_threshold` is too low,
- `top10_threshold` is too low,
- prediction changes frequently,
- one of the attribution metrics does not pass the threshold even though the other does.

Try:

```bash
python run_analysis.py --config config.yaml --cos-threshold 0.3 --top10-threshold 0.2 --plots diagnostics heatmaps
```

### Many plots and long runtime

Use:

```bash
python run_analysis.py --config config.yaml --only-summary
```

or select only specific groups:

```bash
python run_analysis.py --config config.yaml --plots heatmaps rankings
```

---

## 17. Suggested Research Workflow

1. Run everything without plots:

```bash
python run_analysis.py --config config.yaml --no-plots
```

2. Check:

```text
outputs/runs/all_models__all_attacks/csv/02_summary_by_model_attack.csv
```

3. Generate main plots:

```bash
python run_analysis.py --config config.yaml --only-plots --plots heatmaps rankings tradeoffs
```

4. If AASR looks odd, run diagnostics:

```bash
python run_analysis.py --config config.yaml --only-plots --plots diagnostics
```

5. Check fake/real differences:

```bash
python run_analysis.py --config config.yaml --only-plots --plots labels
```

6. For a specific combination, check scores:

```bash
python run_analysis.py --config config.yaml --model AST --attack Attack_1 --plots scores
```

---

## 18. Final Interpretation

A model or model x attack combination is more susceptible to an attribution attack if it has:

```text
high AFS_stable
high AASR
high pred_preserved_rate
low cos_sim
low top10_overlap
high audio quality
low fraction of near_boundary_adv_010
```

Most recommended primary ranking metric:

```text
AFS stable
```

Treat AASR as a hard, threshold-based success metric, and AFS as a continuous attribution fragility metric.

## Visualization and Diagnostics Extensions

The project includes additional modules for deeper analysis of attacks on XAI explanations:

- `distributions` — sample-level metric distributions: boxplots, ECDF, scatter `cos_sim` vs `top10_overlap`.
- `thresholds` — AASR sensitivity analysis over `cos_threshold` and `top10_threshold`.
- `cases` — automatic export of the most interesting samples for inspection.
- `interactive` — HTML dashboard in Plotly, if the `plotly` library is installed.

Example:

```bash
python run_analysis.py --plots all
```

or only the new plots:

```bash
python run_analysis.py --plots distributions thresholds cases interactive
```

A detailed description of changes and motivation can be found in `CHANGELOG_VISUALIZATION_IMPROVEMENTS.md`.

## Individual Sample Analysis

```bash
python analyze_audio_easy_hard.py \
  --audio-root "" \
  --easy-names "path/to/attribution_attack_analysis/selection/easy_sample_names.txt" \
  --hard-names "path/to/attribution_attack_analysis/selection/hard_sample_names.txt" \
  --output-dir "path/to/attribution_attack_analysis/audio_case_analysis" \
  --layout sample_dirs \
  --audio-filename original.wav
```
