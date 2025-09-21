# Tier 1.5 Report (Oblique Linear, k=6)

**Run:** `artifacts/20250921_135009/tier1_k6/`

## Data & Feature Construction
- Attribute generator identical to Tiers 0–1 (10,000 rows, 60 columns, 24 informative with 12 positive-only).
- Oracle features: 20 standardized linear blends, each spanning 6 distinct informative attributes (coverage queue guarantees every informative attribute appears ≥1×).
- Weights sampled from `{±2, ±1, ±0.5}`; coefficients for the logistic head drawn `Uniform(0.5, 1.5)` with random signs, followed by intercept calibration to 10 % positives (`σ=0.5`).

## Training Configuration
- Optuna TPE with 10 trials per arm, wrapped in a 300 s global budget (observed runtime ≈4.7 minutes).
- XGBoost: hist tree method, `max_depth=6`, `n_estimators=1000`, PR-AUC early stopping, `n_jobs=-1`.
- Median imputation, `scale_pos_weight=2` shared between arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.7863 | 0.9592  | 0.1442  | 450            | 31.91       |
| FE-Oracle  | 0.7872 | 0.9625  | 0.1389  | 85             | 9.28        |

## Notes
- Both arms improve over k=3, with FE-Oracle retaining a modest PR-AUC lead while converging in fewer trees (85 vs 450), suggesting higher-arity blends are harder to reconstruct exactly from raw attributes.
- Attribute coverage metadata shows all 24 informative columns participated at least once despite the larger k.
- Runtime stays just inside the 5-minute limit thanks to the reduced Optuna trial budget; per-trial fits remain under 35 s.

## Next
- Move on to Tier 2 (pairwise compositional templates) and generate a corresponding report once the generators and harness are in place.
