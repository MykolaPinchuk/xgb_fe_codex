# Tier 2.3 Report (Pairwise Absolute Differences)

**Run:** `artifacts/20250921_194406/tier2_absdiff/`

## Data & Feature Construction
- Base attributes: 10,000 rows, 60 numeric columns, 24 informative (12 positive-only) drawn from correlated blocks with heterogeneous scales and heavy-tailed distractors.
- Feature equation: $z_m = |x_{i_m} - x_{j_m}|$ with $(i_m, j_m)$ drawn from the informative pool and standardized post-hoc. Coverage queue ensures every informative attribute appears at least once across the feature set.
- Logistic coefficients sampled `Uniform(0.5, 1.5)` with random signs; intercept solved for a 10 % positive rate given `σ=0.5` jitter.

## Training Configuration
- Optuna TPE, 10 trials per arm, global timeout 300 s (observed ≈5.2 minutes).
- XGBoost hist, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation with shared `scale_pos_weight=2`.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.5345 | 0.8898  | 0.2145  | 230            | 1.45        |
| FE-Oracle  | 0.6080 | 0.9144  | 0.1961  | 161            | 0.92        |

## Notes
- Absolute differences still produce a material gulf (ΔPR-AUC ≈ 0.074) with FE-Oracle converging faster, implying ATTR needs deeper interactions to capture varying scales in the correlated backbone.
- Both arms now stop well before the `n_estimators` ceiling (<250 trees), so depth sweeps may be less critical than richer feature engineering for ATTR.
- Attribute coverage remains complete across the informative pool.

## Next
- See `docs/tier2_minmax.md` for min/max extrema results; upcoming work will consolidate Tier 2 metrics and progress to Tier 3 compositions.
