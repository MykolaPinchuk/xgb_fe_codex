# Tier 2.4 Report (Pairwise Min/Max)

**Run:** `artifacts/20250921_143224/tier2_minmax/`

## Data & Feature Construction
- Attribute generator unchanged (10,000 rows, 60 columns, 24 informative inputs with 12 positive-only).
- Feature equations alternate between $z_{2m} = \min(x_{i_m}, x_{j_m})$ and $z_{2m+1} = \max(x_{i_m}, x_{j_m})$, each standardized after construction; a coverage queue guarantees every informative attribute appears ≥1× across the set.
- Logistic head samples coefficients from `Uniform(0.5, 1.5)` with random signs; intercept solved for 10 % prevalence with `σ=0.5` Gaussian jitter.

## Training Configuration
- Optuna TPE (10 trials per arm) under the 300 s cap; observed runtime ≈5.3 minutes.
- XGBoost hist, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation; `scale_pos_weight=2` shared across arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.7150 | 0.9430  | 0.1620  | 610            | 27.45       |
| FE-Oracle  | 0.7989 | 0.9696  | 0.1290  | 525            | 22.06       |

## Notes
- FE-Oracle gains ≈0.084 PR-AUC over ATTR, reflecting the benefit of direct access to order-statistic features that trees approximate only after many splits.
- Both arms run deep into the early-stopping window (≥500 rounds), hinting that shallower/alternative models might struggle with extrema-driven signals.
- Coverage confirms all informative attributes participate despite the alternating min/max generation.

## Next
- Tier 2 now includes products, ratios, abs-diffs, and min/max. Remaining work: aggregate comparisons across specs and advance to Tier 3 multi-variate compositions.
