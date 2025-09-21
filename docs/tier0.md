# Tier 0 Report

**Run:** `artifacts/20250921_131812/tier0/`

## Data & Setup
- Sample size: 10,000 rows, 60 numeric attributes.
- Informative subset: 24 attributes, with 12 constrained positive (exp-transformed).
- Tier 0 features: labels generated from a linear logit over 5 informative columns; oracle features reuse these raw columns.
- Target prevalence: 10% positives via intercept search (`σ=0.5` jitter).

Selected columns: `['x4', 'x18', 'x23', 'x11', 'x5']`
Weights: `[1.4128, -0.5033, -1.3622, 1.2760, -1.4181]`

## Training Configuration
- Optuna TPE, 15 trials per arm (budgeted inside a 300 s run cap; actual total ≈20 s).
- XGBoost hist method, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC (`n_jobs=-1`).
- Class handling: `scale_pos_weight = 2` for both arms.
- Median imputation on train/val splits (80/20 stratified).

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.4903 | 0.8636  | 0.2484  | 64             | 0.723       |
| FE-Oracle  | 0.5074 | 0.8751  | 0.2375  | 290            | 0.647       |

## Notes
- Expected parity holds: FE-Oracle enjoys a slight edge because it focuses only on the contributory columns, while ATTR learns to up-weight the same signals from the 60-column space.
- Best iteration for FE-Oracle ran longer (290 vs 64 rounds) despite higher PR-AUC; likely due to the smaller feature space requiring more trees to reach saturation—worth monitoring as tiers get richer.
- No runtime issues; total experiment duration stayed well below the 5-minute ceiling.

## Next
- Implement Tier 1 generators (oblique linear features) and extend reporting with same template once validated.
