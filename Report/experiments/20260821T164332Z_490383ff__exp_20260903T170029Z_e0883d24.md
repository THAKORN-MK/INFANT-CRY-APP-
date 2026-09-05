# Shared Experiment Comparison

Experiment run: `20260821T164332Z_490383ff__exp_20260903T170029Z_e0883d24`
Wave: `stage1_baselines`
Verified payload SHA-256: `6f084607b65dba0bf7a16f6a267956e57fd126ae97bd0fe6cdf5939f73333cbc`

Evaluation and ranking scope: **corrected grouped OOF only**.
Final Test was unavailable to ranking, model selection, and promotion checks.

## Reference OOF metrics

- `oof_macro_f1`: 0.989114
- `oof_balanced_accuracy`: 0.989680
- `oof_accuracy`: 0.989309
- `minimum_class_recall`: 0.986994

## Verified candidate leaderboard

| Rank | Candidate | Wave | Seeds | OOF Macro F1 | OOF Balanced Accuracy | OOF Accuracy | Minimum class recall | Parameters | Verification |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | stage1_mfcc_svm | stage1_baselines | 42 | 0.975676 | 0.975456 | 0.976151 | 0.970420 | 30408.4 | complete |
| 2 | stage1_logmel_small_cnn | stage1_baselines | 42 | 0.924640 | 0.927892 | 0.925576 | 0.911127 | 10754.0 | complete |
| 3 | stage1_majority | stage1_baselines | 42 | 0.362683 | 0.500000 | 0.569079 | 0.000000 | 0.0 | complete |

## Excluded/failed candidates

- None.

## Per-class OOF comparison

- `stage1_majority`: baby=0.000000, not_baby=1.000000
- `stage1_mfcc_svm`: baby=0.970420, not_baby=0.980491
- `stage1_logmel_small_cnn`: baby=0.944656, not_baby=0.911127

## Stability across seeds

- Single-seed screening; stability confirmation is pending.

## Promotion checks

Status: `no_promotion_recommended`
- FAIL — `three_seed_confirmation`

## Limitations

- Candidate ranking uses corrected grouped internal OOF validation.
- Fold estimates are correlated and are not independent experiments.
- Independent external validation has not yet been performed.
