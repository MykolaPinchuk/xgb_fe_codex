# Implementation Notes

## Current scope (Tiers 0–3)

### Shared setup

- Attribute generator returns 60 numeric columns (`x0`–`x59`) with 24 informative-eligible attributes built from correlated Gaussian blocks (block size ≈6, ρ≈0.4). Columns receive heterogeneous scale multipliers; half of the informative set is warped through a log-normal transform to enforce positivity without collapsing variance. The remaining 36 distractors mix lower-correlation blocks with heavy-tailed draws to supply nuisance structure. Metadata records which indices remain positive-only.
- Logistic label builder samples coefficients `β ~ U(0.5, 1.5)` with random signs, injects Gaussian jitter (`σ=0.5` by default), and solves for an intercept hitting ≈10% positives.
- CLI (`python -m cli.run_experiment --tier ...`) orchestrates Tier selection, Optuna-driven XGBoost training (10 trials per arm, global 300 s budget), and artifact logging under `artifacts/<timestamp>/<tier_run>/`.
- XGBoost runs with `tree_method="hist"`, `max_depth=6`, `n_estimators=1000`, `early_stopping_rounds=100`, and `n_jobs=-1`.
- Summary helper: `python -m cli.summarize_runs --root artifacts --latest-only` prints a consolidated metrics table (see README for details).

### Tier 0

- Labels derive from a linear logit over five informative attributes; the oracle feature matrix simply reuses those raw columns.
- Report: `docs/tier0.md` (run `artifacts/20250921_131812/tier0/`).

### Tier 1 (oblique linear, k ∈ {2,…,6})

- Oracle features: 20 standardized linear blends, each combining `k` informative attributes with weights drawn from `{±2, ±1, ±0.5}`; a coverage queue ensures every informative attribute appears at least once even for higher arities.
- CLI flag `--k` selects the arity (default 3). Oracle labels follow the shared logistic recipe using those features.
- Reports: `docs/tier1.md` for `k=3` (`artifacts/20250921_133826/tier1_k3/`) and `docs/tier1_k6.md` for the high-arity variant (`artifacts/20250921_135009/tier1_k6/`).

### Tier 2 (pairwise compositional — products, ratios, abs-diff, min/max)

- Products: 20 standardized pairwise products with coverage ensuring each informative attribute appears at least once.
- Ratios: 20 standardized ratios with denominator safety (`abs` + `ε`) plus per-feature weights drawn from a wide pool to ensure denominators remain heterogeneous (recorded via `denominator_cv`).
- Abs-diff: 20 standardized absolute differences capturing distance-like signals.
- Min/Max: 20 standardized extrema alternating between `min` and `max` features while maintaining coverage.
- CLI flag `--spec` selects the template (`product`, `ratio`, `absdiff`, `minmax`).
- Reports: `docs/tier2_product.md` (`artifacts/20250921_135908/tier2_product/`), `docs/tier2_ratio.md` (`artifacts/20250921_140729/tier2_ratio/`), `docs/tier2_absdiff.md` (`artifacts/20250921_141419/tier2_absdiff/`), `docs/tier2_minmax.md` (`artifacts/20250921_143224/tier2_minmax/`).

### Tier 3 (multi-variate compositions, k ∈ {3,4})

- Ratio-of-sums: features take the form $(\sum a_i x_i) / (\sum b_j |x_j| + \varepsilon)$ with numerator size 1–2 and denominator size 2–3 (total arity 3 or 4). Denominator weights come from a widened pool and are resampled until the coefficient of variation clears a configurable threshold, preventing near-constant denominators.
- Sum-product mixes: features follow $(\sum a_i x_i) \times x_k$ or $(\sum a_i x_i) \times (\sum c_j x_j)$, keeping total arity at 3–4 while alternating between single and multi-variate factors.
- CLI usage: `python -m cli.run_experiment --tier tier3 --spec ratioofsum --k 4` or `--spec sumproduct --k 4` (default `k=3`).
- Reports: `docs/tier3_ratioofsum.md` (`artifacts/20250921_182613/tier3_ratioofsum_k4/`), `docs/tier3_sumproduct.md` (`artifacts/20250921_183013/tier3_sumproduct_k4/`).

### Tier 4 (high-arity compositions, k ∈ {5,6})

- Ratio-of-sums: numerator groups of size 2–3 divided by absolute-value denominators covering the remaining attributes (total arity 5 or 6) with `ε = 10^{-3}` for stability. Denominator weights are re-drawn until the coefficient of variation exceeds the configured minimum, restoring difficulty at high arity.
- Sum-product mixes: combinations such as $(\sum a_i x_i) \times x_k$ (with large left sums) or $(\sum a_i x_i) \times (\sum c_j x_j)$ where each factor spans ≥2 informative attributes.
- CLI usage: `python -m cli.run_experiment --tier tier4 --spec ratioofsum --k 6` or `--spec sumproduct --k 6` (default `k=5`).
- Reports: `docs/tier4_ratioofsum.md` (`artifacts/20250921_183528/tier4_ratioofsum_k6/`), `docs/tier4_sumproduct.md` (`artifacts/20250921_183727/tier4_sumproduct_k6/`).

## Near-term roadmap

1. Rerun Tier 0–4 experiments so artifacts reflect the correlated/mixed-scale attribute backbone.
2. Add tier-aware feature constructors, evaluation tests, and richer reporting.
3. Introduce tier-specific property tests to guard data quality and training behavior.

This document will evolve as new tiers and capabilities are implemented.
