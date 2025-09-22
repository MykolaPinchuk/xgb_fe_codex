# Tier 2.1 Report (Pairwise Products)

**Run:** `artifacts/20250921_194304/tier2_product/`

## Data & Feature Construction
- Attribute generator matches the upgraded Tier 0 backbone (10,000 rows, 60 numeric columns, 24 informative candidates drawn from correlated blocks with log-normal positive-only transforms and heavy-tailed distractors).
- Feature equation: $z_m = x_{i_m} \cdot x_{j_m}$ with $(i_m, j_m)$ drawn from the informative pool and standardized to zero mean / unit variance. A coverage queue ensures every informative attribute appears in ≥1 product despite the pairwise restriction.
- Logistic head uses coefficients drawn `Uniform(0.5, 1.5)` with random signs; intercept solved for a 10 % positive rate with Gaussian jitter (`σ=0.5`).

## Training Configuration
- Optuna TPE with 10 trials per arm inside the 300 s global cap (observed duration ≈4.9 minutes).
- XGBoost: histogram tree method, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation and `scale_pos_weight=2` shared across arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.5159 | 0.8606  | 0.2408  | 105            | 1.32        |
| FE-Oracle  | 0.6537 | 0.9125  | 0.1978  | 785            | 2.31        |

## Notes
- FE-Oracle now leads by ≈0.138 PR-AUC, underlining how multiplicative structure becomes significantly harder for ATTR once attribute scales and correlations diverge.
- Coverage logs confirm every informative attribute appears in ≥1 product; diagnostics show informative variances spanning `0.49–30.7` with moderate inter-factor correlations.
- ATTR typically stops near 100 trees while FE-Oracle pushes deeper (~785 trees) as it fits the standardized product features directly.

## Next
- Extend Tier 2 to other pairwise templates (ratios, abs-diffs, mins/maxes) before moving up to Tier 3 multi-variate compositions.
