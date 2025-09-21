# Implementation Notes

## Current scope (Tiers 0–1)

### Shared setup

- Attribute generator returns 60 numeric columns (`x0`–`x59`) with 24 informative-eligible attributes (12 positive-only via exp transform) and 36 distractors.
- Logistic label builder samples coefficients `β ~ U(0.5, 1.5)` with random signs, injects Gaussian jitter (`σ=0.5` by default), and solves for an intercept hitting ≈10% positives.
- CLI (`python -m cli.run_experiment --tier ...`) orchestrates Tier selection, Optuna-driven XGBoost training (10 trials per arm, global 300 s budget), and artifact logging under `artifacts/<timestamp>/<tier_run>/`.
- XGBoost runs with `tree_method="hist"`, `max_depth=6`, `n_estimators=1000`, `early_stopping_rounds=100`, and `n_jobs=-1`.

### Tier 0

- Labels derive from a linear logit over five informative attributes; the oracle feature matrix simply reuses those raw columns.
- Report: `docs/tier0.md` (run `artifacts/20250921_131812/tier0/`).

### Tier 1 (oblique linear, k ∈ {2,…,6})

- Oracle features: 20 standardized linear blends, each combining `k` informative attributes with weights drawn from `{±2, ±1, ±0.5}`; a coverage queue ensures every informative attribute appears at least once even for higher arities.
- CLI flag `--k` selects the arity (default 3). Oracle labels follow the shared logistic recipe using those features.
- Reports: `docs/tier1.md` for `k=3` (`artifacts/20250921_133826/tier1_k3/`) and `docs/tier1_k6.md` for the high-arity variant (`artifacts/20250921_135009/tier1_k6/`).

## Near-term roadmap

1. Flesh out generators for higher tiers (oblique linear, pairwise compositions, etc.).
2. Add tier-aware feature constructors, evaluation tests, and richer reporting.
3. Introduce tier-specific property tests to guard data quality and training behavior.

This document will evolve as new tiers and capabilities are implemented.
