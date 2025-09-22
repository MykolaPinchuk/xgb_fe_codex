# Tier 2.2 Report (Pairwise Ratios)

**Run:** `artifacts/20250921_140729/tier2_ratio/`

## Data & Feature Construction
- Same attribute design as earlier tiers (10,000 rows, 60 numeric columns, 24 informative with half positive-only).
- Feature equation: $z_m = \dfrac{x_{i_m}}{|x_{j_m}| + \varepsilon}$ (with $\varepsilon = 10^{-3}$), falling back to raw denominators when $x_{j_m}$ is from the positive-only subset. Coverage ensures every informative attribute appears as numerator or denominator at least once.
- Logistic head samples coefficients from `Uniform(0.5, 1.5)` with random signs; intercept chosen for a 10 % positive rate given `σ=0.5` jitter.

## Training Configuration
- Optuna TPE with 10 trials per arm, global timeout 300 s (observed duration ≈4.6 minutes).
- XGBoost: histogram tree method, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation; `scale_pos_weight=2` shared by both arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.5904 | 0.9022  | 0.2092  | 351            | 16.43       |
| FE-Oracle  | 0.6664 | 0.9365  | 0.1731  | 254            | 14.09       |

## Notes
- Ratios widen the ATTR vs FE-Oracle gap further (ΔPR-AUC ≈ 0.076), underscoring the challenge for trees to reconstruct division, especially when denominators span both positive-only and signed attributes.
- Attribute usage logs confirm full coverage despite the higher risk of duplicated denominators.
- Early stopping remains effective: both arms halt well before 1000 trees, keeping per-trial fits under ~17 s.

## Next
- Implement the remaining Tier 2 templates (abs-diff, min/max) to complete the pairwise feature suite before moving on to Tier 3 compositions.
