# Tier 1 Report (Oblique Linear, k=3)

**Run:** `artifacts/20250921_133826/tier1_k3/`

## Data & Feature Construction
- Same attribute backbone as Tier 0 (10,000 rows by default; run sampled at full scale).
- Informative pool: 24 attributes, evenly reused via coverage pass (no unused informative columns in this run).
- Each of the 20 oracle features is a linear blend of exactly 3 informative attributes with weights drawn from `{-2, -1, -0.5, 0.5, 1, 2}` and standardized to zero mean / unit variance.
- Logistic layer: coefficients sampled `Uniform(0.5, 1.5)` with random signs, intercept solved to yield 10 % positives.

## Training Configuration
- Optuna TPE with 10 trials per arm; global timeout 300 s (actual run ≈40 s).
- XGBoost hist, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation; identical hyperparameter ranges and imbalance handling (`scale_pos_weight=2`).

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.7006 | 0.9437  | 0.1692  | 329            | 1.94        |
| FE-Oracle  | 0.7140 | 0.9528  | 0.1560  | 486            | 2.26        |

## Notes
- FE-Oracle widens the PR-AUC gap (~0.013) relative to Tier 0, reflecting the benefit of direct access to the linear blends instead of recovering them from the 60-attr space.
- Coverage metadata confirms all 24 informative attributes fed at least one feature.
- Early stopping still triggers well below the `n_estimators` ceiling (≤500 trees), keeping each trial fast enough for the 5-minute global constraint.

## Next
- Implement Tier 2 feature constructors (pairwise compositional templates) and capture analogous reports once validated.
