# Tier 2.2 Report (Pairwise Ratios)

**Run:** `artifacts/20250921_194334/tier2_ratio/`

## Data & Feature Construction
- Same correlated/mixed-scale attribute design as the refreshed Tier 0 (10,000 rows, 60 numeric columns, 24 informative with half positive-only and heterogeneous scales).
- Feature equation: $z_m = \dfrac{x_{i_m}}{|x_{j_m}| + \varepsilon}$ (with $\varepsilon = 10^{-3}$), falling back to raw denominators when $x_{j_m}$ is from the positive-only subset. Coverage ensures every informative attribute appears as numerator or denominator at least once.
- Logistic head samples coefficients from `Uniform(0.5, 1.5)` with random signs; intercept chosen for a 10 % positive rate given `σ=0.5` jitter.

## Training Configuration
- Optuna TPE with 10 trials per arm, global timeout 300 s (observed duration ≈4.6 minutes).
- XGBoost: histogram tree method, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation; `scale_pos_weight=2` shared by both arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.6291 | 0.8926  | 0.2138  | 224            | 1.36        |
| FE-Oracle  | 0.7265 | 0.9336  | 0.1827  | 126            | 0.51        |

## Notes
- ATTR now trails FE-Oracle by ≈0.097 PR-AUC even though both arms stop well before 300 trees, signalling that the weighted denominators force trees to learn sharper non-linearities.
- Denominator diagnostics show substantial spread (medians between ~0.70 and 1.58 with 99th percentiles up to ~20.6), confirming the mixed-scale weighting keeps denominators from collapsing.
- Attribute usage logs still confirm full coverage of all informative columns for both numerators and denominators.

## Next
- Implement the remaining Tier 2 templates (abs-diff, min/max) to complete the pairwise feature suite before moving on to Tier 3 compositions.
