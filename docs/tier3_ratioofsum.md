# Tier 3.1 Report (Ratio-of-Sums, k=4)

**Run:** `artifacts/20250921_194521/tier3_ratioofsum_k4/`

## Data & Feature Construction
- Attributes follow the upgraded Meta-Iteration 1 backbone (10,000 rows, 60 columns, 24 informative candidates with correlated/mixed-scale signals and log-normal positive-only columns).
- Feature equation: $z_m = \dfrac{\sum_{j=1}^{s_m} a_{m,j} x_{i_{m,j}}}{\sum_{r=1}^{t_m} b_{m,r} |x_{u_{m,r}}| + \varepsilon}$ with $s_m \in \{1,2\}$, $t_m \in \{2,3\}$, $s_m + t_m = 4$, $a_{m,j} \in \{-2, -1, -0.5, 0.5, 1, 2\}$, $b_{m,r} \in \{0.5, 1, 2\}$, and $\varepsilon = 10^{-3}$. Coverage queues ensure every informative attribute appears in at least one numerator or denominator.
- Logistic layer samples coefficients from `Uniform(0.5, 1.5)` with random signs and calibrates the intercept to 10 % prevalence under Gaussian jitter (`σ=0.5`).

## Training Configuration
- Optuna TPE with 10 trials per arm under the 300 s global cap (observed runtime ≈40 s thanks to early convergence).
- XGBoost histogram method, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation applied per split; `scale_pos_weight=2` shared between arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.6326 | 0.9200  | 0.1968  | 778            | 3.60        |
| FE-Oracle  | 0.7804 | 0.9595  | 0.1446  | 885            | 2.24        |

## Notes
- FE-Oracle now outpaces ATTR by ≈0.148 PR-AUC, still a substantial gap that proves the weighted denominators remain challenging even though ATTR improved under the new sampling scheme.
- Diagnostics show denominator medians spanning roughly `1.1–7.6` with heavy upper tails (p99 up to ~45) and logged coefficients of variation typically >0.4, confirming the resampling guardrails keep denominators from collapsing.
- Attribute usage logs confirm the numerator/denominator coverage requirement, with no informative columns left unused.

## Next
- Implement the complementary Tier 3 sum-product template, then compare how ATTR vs FE-Oracle gaps differ between multiplicative numerators and denominator-heavy ratios.
