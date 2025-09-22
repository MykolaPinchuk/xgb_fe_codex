# Tier 2.3 Report (Pairwise Absolute Differences)

**Run:** `artifacts/20250921_141419/tier2_absdiff/`

## Data & Feature Construction
- Base attributes: 10,000 rows, 60 numeric columns, 24 informative (12 positive-only) as in earlier tiers.
- Feature equation: $z_m = |x_{i_m} - x_{j_m}|$ with $(i_m, j_m)$ drawn from the informative pool and standardized post-hoc. Coverage queue ensures every informative attribute appears at least once across the feature set.
- Logistic coefficients sampled `Uniform(0.5, 1.5)` with random signs; intercept solved for a 10 % positive rate given `σ=0.5` jitter.

## Training Configuration
- Optuna TPE, 10 trials per arm, global timeout 300 s (observed ≈5.2 minutes).
- XGBoost hist, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation with shared `scale_pos_weight=2`.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.6892 | 0.9379  | 0.1766  | 571            | 27.05       |
| FE-Oracle  | 0.7961 | 0.9635  | 0.1344  | 622            | 32.97       |

## Notes
- Absolute differences produce a large gulf between FE-Oracle and ATTR (ΔPR-AUC ≈ 0.11), consistent with trees needing deeper interactions to model sharp distance cues.
- Both arms run close to the early-stopping ceiling (>500 rounds), suggesting further depth/feature sweeps may be warranted for this spec.
- Attribute coverage verified: all 24 informative attributes participate in at least one absolute-difference feature.

## Next
- See `docs/tier2_minmax.md` for min/max extrema results; upcoming work will consolidate Tier 2 metrics and progress to Tier 3 compositions.
