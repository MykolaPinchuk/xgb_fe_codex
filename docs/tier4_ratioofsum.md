# Tier 4 Report (Ratio-of-Sums, k=6)

**Run:** `artifacts/20250921_183528/tier4_ratioofsum_k6/`

## Data & Feature Construction
- Attributes follow the Meta-Iteration 1 default (10,000 rows, 60 numeric columns, 24 informative with 12 positive-only).
- Feature equation: $z_m = \dfrac{\sum_{j=1}^{s_m} a_{m,j} x_{i_{m,j}}}{\sum_{r=1}^{t_m} b_{m,r} |x_{u_{m,r}}| + \varepsilon}$ with $s_m \in \{2,3\}$, $t_m = 6 - s_m \geq 2$, coefficients $a_{m,j} \in \{-2, -1, -0.5, 0.5, 1, 2\}$, $b_{m,r} \in \{0.5, 1, 2\}$, and $\varepsilon = 10^{-3}$. Coverage queues ensure every informative attribute appears as numerator or denominator at least once across the 15 oracle features.
- Logistic layer samples coefficients from `Uniform(0.5, 1.5)` with random signs and calibrates the intercept to 10 % prevalence under Gaussian jitter (`σ=0.5`).

## Training Configuration
- Optuna TPE with 10 trials per arm (global 300 s cap, observed runtime ≈40 s).
- XGBoost histogram method, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation per split; `scale_pos_weight=2` shared across arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.6446 | 0.9202  | 0.1867  | 354            | 2.53        |
| FE-Oracle  | 0.7236 | 0.9444  | 0.1631  | 997            | 3.85        |

## Notes
- FE-Oracle enjoys a PR-AUC lead of ≈0.079 despite both arms pushing toward the 1000-tree ceiling, signalling the difficulty of reconstructing high-arity ratios from raw attributes.
- Oracle arm nearly exhausts the estimator budget (best iteration 997); deeper trees or expanded Optuna budgets may be worth exploring in later iterations for Tier 4 stability.
- Coverage metadata confirms all 24 informative attributes appear in the numerator or denominator at least once.

## Next
- Run the Tier 4 sum-product variant to compare against ratio-of-sums, then prepare aggregate Tier 3/4 summaries ahead of Meta-Iteration 2 enhancements.
