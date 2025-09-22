# Tier 1.5 Report (Oblique Linear, k=6)

**Run:** `artifacts/20250921_194233/tier1_k6/`

## Data & Feature Construction
- Attribute generator identical to the updated Tier 0 setup (10,000 rows, 60 columns, 24 informative across correlated blocks with positive-only log-normal transforms on half the pool).
- Feature equation: $z_m = \sum_{j=1}^{6} w_{m,j} \cdot x_{i_{m,j}}$ with $w_{m,j} \in \{-2, -1, -0.5, 0.5, 1, 2\}$ and each feature using 6 distinct informative attributes (coverage queue guarantees every informative attribute appears ≥1×).
- Logistic head coefficients drawn `Uniform(0.5, 1.5)` with random signs, followed by intercept calibration to 10 % positives (`σ=0.5`).

## Training Configuration
- Optuna TPE with 10 trials per arm, wrapped in a 300 s global budget (observed runtime ≈4.7 minutes).
- XGBoost: hist tree method, `max_depth=6`, `n_estimators=1000`, PR-AUC early stopping, `n_jobs=-1`.
- Median imputation, `scale_pos_weight=2` shared between arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.5732 | 0.9054  | 0.2067  | 162            | 1.08        |
| FE-Oracle  | 0.5769 | 0.9135  | 0.2082  | 56             | 0.58        |

## Notes
- Higher-arity blends now show only a slight PR-AUC gap (~0.004) but FE-Oracle still converges roughly 3× faster (56 vs 162 trees), indicating attribute correlations make some k=6 combinations easier to approximate.
- Coverage metadata shows all 24 informative columns participated at least once; attribute diagnostics mirror Tier 1 with variances spanning `0.49–30.7` and modest cross-correlations.
- With the richer backbone, per-trial fits finish in ~1 s, leaving ample headroom inside the 300 s global Optuna budget.

## Next
- Move on to Tier 2 (pairwise compositional templates) and generate a corresponding report once the generators and harness are in place.
