# Tier 4 Report (Sum-Product, k=6)

**Run:** `artifacts/20250921_183727/tier4_sumproduct_k6/`

## Data & Feature Construction
- Attributes follow the Meta-Iteration 1 default (10,000 rows, 60 numeric columns, 24 informative with 12 positive-only).
- Feature equation: $z_m = \left(\sum_{j=1}^{s_m} a_{m,j} x_{i_{m,j}}\right) \times \begin{cases} c_{m,1} x_{u_{m,1}} & \text{if } t_m = 1, \\ \sum_{r=1}^{t_m} c_{m,r} x_{u_{m,r}} & \text{if } t_m \ge 2 \end{cases}$ with $s_m + t_m = 6$, $s_m \in \{4,5\}$ when paired with a single factor and $s_m \in \{3,4\}$ when both factors are sums. Coefficients draw from $\{-2, -1, -0.5, 0.5, 1, 2\}$ and are sampled with replacement. Coverage queues guarantee every informative attribute appears in at least one factor across the 15 oracle features.
- Logistic layer samples coefficients from `Uniform(0.5, 1.5)` with random signs and calibrates the intercept to 10 % prevalence (`σ=0.5` jitter).

## Training Configuration
- Optuna TPE with 10 trials per arm (global 300 s cap, observed runtime ≈25 s).
- XGBoost histogram method, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation per split; `scale_pos_weight=2` shared across arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.6807 | 0.9234  | 0.1874  | 197            | 1.54        |
| FE-Oracle  | 0.7936 | 0.9602  | 0.1474  | 109            | 0.50        |

## Notes
- FE-Oracle widens the PR-AUC gap to ≈0.113 while converging with far fewer trees (109 vs 197), underscoring how explicit access to the high-arity sum-product factors improves both accuracy and efficiency.
- ATTR performance improves relative to the ratio-of-sums run but still trails the oracle substantially; deeper trees or additional regularisation sweeps might help but will increase runtime.
- Coverage metadata confirms both factors collectively touch the entire informative set.

## Next
- Consolidate Tier 4 metrics (ratios vs sum-products) and update summary tooling before kicking off Meta-Iteration 2 enhancements.
