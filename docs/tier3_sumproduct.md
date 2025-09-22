# Tier 3.2 Report (Sum-Product Mixes, k=4)

**Run:** `artifacts/20250921_194601/tier3_sumproduct_k4/`

## Data & Feature Construction
- Attributes follow the upgraded Meta-Iteration 1 default (10,000 rows, 60 columns, 24 informative with heterogeneous scales and log-normal positive-only transforms).
- Feature equation: $z_m = \left(\sum_{j=1}^{s_m} a_{m,j} x_{i_{m,j}}\right) \times \begin{cases} c_{m,1} x_{u_{m,1}} & \text{if } t_m = 1, \\ \sum_{r=1}^{t_m} c_{m,r} x_{u_{m,r}} & \text{if } t_m \in \{2,3\} \end{cases}$ with total arity $s_m + t_m = 4$, coefficients $a_{m,j}, c_{m,r} \in \{-2, -1, -0.5, 0.5, 1, 2\}$ sampled with replacement, and standardized after construction. Coverage queues ensure every informative attribute participates in at least one factor.
- Logistic layer samples coefficients from `Uniform(0.5, 1.5)` with random signs and calibrates the intercept to a 10 % positive rate under Gaussian jitter (`σ=0.5`).

## Training Configuration
- Optuna TPE with 10 trials per arm within the 300 s budget (observed runtime ≈39 s as early stopping kicks in).
- XGBoost histogram method, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation per split; `scale_pos_weight=2` shared by both arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.7569 | 0.9492  | 0.1547  | 278            | 1.93        |
| FE-Oracle  | 0.7964 | 0.9570  | 0.1427  | 263            | 1.44        |

## Notes
- FE-Oracle now holds a ≈0.040 PR-AUC lead, smaller than the ratio-of-sums gap but still meaningful given identical training budgets.
- Both arms stop well before the `n_estimators` ceiling, and convergence times remain low (<2 s per trial) despite the higher-order compositions.
- Attribute usage logs confirm both left and right factors cover the informative pool at least once.

## Next
- Consolidate Tier 3 metrics (ratios vs sum-products) and prepare for Tier 4 high-arity extensions once Meta-Iteration 1 scope allows.
