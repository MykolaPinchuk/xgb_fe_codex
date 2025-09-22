# Tier 3.2 Report (Sum-Product Mixes, k=4)

**Run:** `artifacts/20250921_183013/tier3_sumproduct_k4/`

## Data & Feature Construction
- Attributes follow the Meta-Iteration 1 default (10,000 rows, 60 columns, 24 informative with 12 positive-only).
- Feature equation: $z_m = \left(\sum_{j=1}^{s_m} a_{m,j} x_{i_{m,j}}\right) \times \begin{cases} c_{m,1} x_{u_{m,1}} & \text{if } t_m = 1, \\ \sum_{r=1}^{t_m} c_{m,r} x_{u_{m,r}} & \text{if } t_m \in \{2,3\} \end{cases}$ with total arity $s_m + t_m = 4$, coefficients $a_{m,j}, c_{m,r} \in \{-2, -1, -0.5, 0.5, 1, 2\}$ sampled with replacement, and standardized after construction. Coverage queues ensure every informative attribute participates in at least one factor.
- Logistic layer samples coefficients from `Uniform(0.5, 1.5)` with random signs and calibrates the intercept to a 10 % positive rate under Gaussian jitter (`σ=0.5`).

## Training Configuration
- Optuna TPE with 10 trials per arm within the 300 s budget (observed runtime ≈39 s as early stopping kicks in).
- XGBoost histogram method, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation per split; `scale_pos_weight=2` shared by both arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.7050 | 0.9453  | 0.1733  | 516            | 3.62        |
| FE-Oracle  | 0.7838 | 0.9652  | 0.1376  | 183            | 1.07        |

## Notes
- FE-Oracle leads ATTR by ≈0.079 PR-AUC, mirroring the ratio-of-sums gap and highlighting the challenge trees face when reconstructing nested linear interactions without engineered features.
- Oracle arm converges in fewer trees (183 vs 516) despite higher PR-AUC, indicating direct access to the compositional features boosts efficiency as well as accuracy.
- Attribute usage logs confirm both left and right factors cover the informative pool at least once.

## Next
- Consolidate Tier 3 metrics (ratios vs sum-products) and prepare for Tier 4 high-arity extensions once Meta-Iteration 1 scope allows.
