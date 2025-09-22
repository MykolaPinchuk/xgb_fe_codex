# Tier 1 Report (Oblique Linear, k=3)

**Run:** `artifacts/20250921_194152/tier1_k3/`

## Data & Feature Construction
- Same correlated/mixed-scale attribute backbone as Tier 0 (10,000 rows by default; run sampled at full scale).
- Informative pool: 24 attributes, evenly reused via coverage pass (no unused informative columns in this run).
- Feature equation: $z_m = \sum_{j=1}^{3} w_{m,j} \cdot x_{i_{m,j}}$ with $w_{m,j} \in \{-2, -1, -0.5, 0.5, 1, 2\}$, standardized to zero mean / unit variance.
- Logistic layer: coefficients sampled `Uniform(0.5, 1.5)` with random signs, intercept solved to yield 10 % positives.

## Training Configuration
- Optuna TPE with 10 trials per arm; global timeout 300 s (actual run ≈40 s).
- XGBoost hist, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation; identical hyperparameter ranges and imbalance handling (`scale_pos_weight=2`).

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.5675 | 0.8949  | 0.2134  | 433            | 2.60        |
| FE-Oracle  | 0.5993 | 0.8970  | 0.2090  | 200            | 1.09        |

## Notes
- FE-Oracle now leads by ≈0.032 PR-AUC while converging in less than half the trees, showing how oblique blends become harder to emulate once attribute scales and correlations vary.
- Coverage metadata confirms all 24 informative attributes fed at least one feature; attribute diagnostics report informative variances between `0.49` and `30.7` with mean absolute correlations around `0.08`.
- Early stopping continues to fire well before the `n_estimators` cap, keeping Optuna trials within the 5-minute global window.

## Next
- Implement Tier 2 feature constructors (pairwise compositional templates) and capture analogous reports once validated.
