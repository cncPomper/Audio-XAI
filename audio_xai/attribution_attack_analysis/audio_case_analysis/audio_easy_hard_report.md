# Audio analysis: easy vs hard samples

This report compares audio features for globally easy and hard attribution-attack samples.

Hard samples are the ones previously selected as attribution-stable: prediction can be preserved, but the attribution map does not change easily.

## Counts

- Easy audio files processed: 10
- Hard audio files processed: 10
- Missing audio files: 0

## Top differentiating audio features

- `spectral_bandwidth_mean`: higher for easy; easy_mean=1128, hard_mean=762.6, Cohen_d=2.194, Mann-Whitney p=0.000769
- `dynamic_range_db`: higher for hard; easy_mean=11.53, hard_mean=19.79, Cohen_d=-1.960, Mann-Whitney p=0.00131
- `zcr_mean`: higher for easy; easy_mean=0.1442, hard_mean=0.1075, Cohen_d=1.487, Mann-Whitney p=0.00728
- `high_band_energy_ratio`: higher for easy; easy_mean=0.03589, hard_mean=0.01776, Cohen_d=1.195, Mann-Whitney p=0.0113
- `spectral_flatness_std`: higher for hard; easy_mean=0.05463, hard_mean=0.1011, Cohen_d=-1.138, Mann-Whitney p=0.0312
- `spectral_rolloff85_mean`: higher for easy; easy_mean=1468, hard_mean=1090, Cohen_d=0.997, Mann-Whitney p=0.0539
- `spectral_flux_std`: higher for hard; easy_mean=0.1811, hard_mean=0.2226, Cohen_d=-0.901, Mann-Whitney p=0.14
- `spectral_centroid_mean`: higher for easy; easy_mean=782.3, hard_mean=640.9, Cohen_d=0.871, Mann-Whitney p=0.0539
- `frame_rms_db_std`: higher for hard; easy_mean=6.817, hard_mean=11.52, Cohen_d=-0.853, Mann-Whitney p=0.0173
- `silence_ratio`: higher for hard; easy_mean=0.008854, hard_mean=0.02023, Cohen_d=-0.849, Mann-Whitney p=0.0369

## Interpretation guide

- Large absolute Cohen's d suggests a feature that separates easy and hard samples strongly.
- Positive Cohen's d means the feature is higher for easy samples.
- Negative Cohen's d means the feature is higher for hard samples.
- With only top 10 vs top 10, p-values are exploratory, not definitive.
- Treat this as hypothesis generation: use it to identify patterns worth checking on a larger set.
