# Tier 2.4 Report (Pairwise Min/Max)

**Run:** `artifacts/20250921_194444/tier2_minmax/`

## Data & Feature Construction
- Attribute generator follows the upgraded correlated/mixed-scale design (10,000 rows, 60 columns, 24 informative inputs with 12 positive-only after log-normal transforms).
- Feature equations alternate between $z_{2m} = \min(x_{i_m}, x_{j_m})$ and $z_{2m+1} = \max(x_{i_m}, x_{j_m})$, each standardized after construction; a coverage queue guarantees every informative attribute appears ≥1× across the set.
- Logistic head samples coefficients from `Uniform(0.5, 1.5)` with random signs; intercept solved for 10 % prevalence with `σ=0.5` Gaussian jitter.

## Training Configuration
- Optuna TPE (10 trials per arm) under the 300 s cap; observed runtime ≈5.3 minutes.
- XGBoost hist, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation; `scale_pos_weight=2` shared across arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.6343 | 0.9157  | 0.2063  | 470            | 2.28        |
| FE-Oracle  | 0.6722 | 0.9260  | 0.1886  | 480            | 1.50        |

## Notes
- FE-Oracle holds a ≈0.038 PR-AUC lead while converging in a comparable number of trees, suggesting extrema remain somewhat learnable even under heterogeneous attribute scales.
- Neither arm hits the estimator ceiling; early stopping fires around 470–480 trees, keeping Optuna trials fast (<3 s per fit).
- Coverage confirms all informative attributes participate despite the alternating min/max generation.

## Next
- Tier 2 now includes products, ratios, abs-diffs, and min/max. Remaining work: aggregate comparisons across specs and advance to Tier 3 multi-variate compositions.
