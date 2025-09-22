# Tier 0 Report

**Run:** `artifacts/20250921_194128/tier0/`

## Data & Setup
- Sample size: 10,000 rows, 60 numeric attributes.
- Informative subset: 24 attributes drawn from correlated Gaussian blocks (block size ≈6, ρ≈0.4) with heterogeneous scales; half of the pool is log-normalised to enforce positivity while keeping variance high. The remaining 36 columns mix lower-correlation blocks with heavy-tailed distractors.
- Feature equation: $z_m = x_{i_m}$ (oracle reuses the selected raw columns) and the logit is $f = \sum_{m} \beta_m z_m$ before intercept calibration.
- Target prevalence: 10% positives via intercept search (`σ=0.5` jitter).

Selected columns: `['x3', 'x20', 'x19', 'x23', 'x9']`
Weights: `[-1.0163, 0.7872, -0.5815, 1.0093, -1.1585]`

## Training Configuration
- Optuna TPE, 15 trials per arm (budgeted inside a 300 s run cap; actual total ≈20 s).
- XGBoost hist method, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC (`n_jobs=-1`).
- Class handling: `scale_pos_weight = 2` for both arms.
- Median imputation on train/val splits (80/20 stratified).

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.5280 | 0.8641  | 0.2383  | 147            | 1.41        |
| FE-Oracle  | 0.5398 | 0.8741  | 0.2422  | 57             | 0.34        |

## Notes
- FE-Oracle maintains a modest PR-AUC edge (~0.012) and converges in fewer trees now that the attribute space is richer but still linearly recoverable for Tier 0.
- Attribute diagnostics confirm the correlated backbone: informative variances span `0.48–30.7` with mean absolute correlations around `0.08`, and positive-only columns show broadened spread (median variance ≈7.8).
- No runtime issues; total experiment duration stayed well below the 5-minute ceiling.

## Next
- Implement Tier 1 generators (oblique linear features) and extend reporting with same template once validated.
