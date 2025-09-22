# Tier 2.1 Report (Pairwise Products)

**Run:** `artifacts/20250921_135908/tier2_product/`

## Data & Feature Construction
- Attribute generator matches earlier tiers (10,000 rows, 60 numeric columns, 24 informative candidates with 12 positive-only).
- Feature equation: $z_m = x_{i_m} \cdot x_{j_m}$ with $(i_m, j_m)$ drawn from the informative pool and standardized to zero mean / unit variance. A coverage queue ensures every informative attribute appears in ≥1 product despite the pairwise restriction.
- Logistic head uses coefficients drawn `Uniform(0.5, 1.5)` with random signs; intercept solved for a 10 % positive rate with Gaussian jitter (`σ=0.5`).

## Training Configuration
- Optuna TPE with 10 trials per arm inside the 300 s global cap (observed duration ≈4.9 minutes).
- XGBoost: histogram tree method, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation and `scale_pos_weight=2` shared across arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.6652 | 0.9263  | 0.1846  | 297            | 19.47       |
| FE-Oracle  | 0.7329 | 0.9462  | 0.1599  | 311            | 14.25       |

## Notes
- FE-Oracle enjoys a sizeable PR-AUC advantage (≈0.068) over ATTR, highlighting the difficulty of reconstructing multiplicative structure from raw attributes with the same tree depth.
- Attribute coverage metadata confirms all 24 informative attributes were used as either factor in at least one product.
- Both arms still converge well under the `n_estimators` ceiling; Oracle’s best iteration slightly exceeds ATTR’s, reflecting the added complexity of the engineered features.

## Next
- Extend Tier 2 to other pairwise templates (ratios, abs-diffs, mins/maxes) before moving up to Tier 3 multi-variate compositions.
