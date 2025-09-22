# Tier 3.1 Report (Ratio-of-Sums, k=4)

**Run:** `artifacts/20250921_182613/tier3_ratioofsum_k4/`

## Data & Feature Construction
- Attributes follow the Meta-Iteration 1 default (10,000 rows, 60 columns, 24 informative candidates with 12 positive-only).
- Feature equation: $z_m = \dfrac{\sum_{j=1}^{s_m} a_{m,j} x_{i_{m,j}}}{\sum_{r=1}^{t_m} b_{m,r} |x_{u_{m,r}}| + \varepsilon}$ with $s_m \in \{1,2\}$, $t_m \in \{2,3\}$, $s_m + t_m = 4$, $a_{m,j} \in \{-2, -1, -0.5, 0.5, 1, 2\}$, $b_{m,r} \in \{0.5, 1, 2\}$, and $\varepsilon = 10^{-3}$. Coverage queues ensure every informative attribute appears in at least one numerator or denominator.
- Logistic layer samples coefficients from `Uniform(0.5, 1.5)` with random signs and calibrates the intercept to 10 % prevalence under Gaussian jitter (`σ=0.5`).

## Training Configuration
- Optuna TPE with 10 trials per arm under the 300 s global cap (observed runtime ≈40 s thanks to early convergence).
- XGBoost histogram method, `max_depth=6`, `n_estimators=1000`, early stopping on PR-AUC, `n_jobs=-1`.
- Median imputation applied per split; `scale_pos_weight=2` shared between arms.

## Metrics

| Arm        | PR-AUC | ROC-AUC | Logloss | Best Iteration | Fit Seconds |
|------------|-------:|--------:|--------:|---------------:|------------:|
| ATTR       | 0.7094 | 0.9455  | 0.1629  | 710            | 4.30        |
| FE-Oracle  | 0.7887 | 0.9593  | 0.1385  | 797            | 3.45        |

## Notes
- FE-Oracle leads ATTR by ≈0.079 PR-AUC, underscoring how difficult it is for trees to approximate stacked ratios without explicit engineered features.
- Both arms run deep into the early-stopping window (>700 trees), indicating these compositions are more intricate than Tier 2 pairwise templates.
- Attribute usage logs confirm the numerator/denominator coverage requirement, with no informative columns left unused.

## Next
- Implement the complementary Tier 3 sum-product template, then compare how ATTR vs FE-Oracle gaps differ between multiplicative numerators and denominator-heavy ratios.
