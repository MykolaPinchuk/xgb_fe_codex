# Implementation Notes

## Current scope (Tier 0)

- Generate 60 numeric attributes (`x0`–`x59`).
  - 24 attributes are marked as informative-eligible but remain independent Gaussians.
  - The remaining attributes act as distractors with similar distributions.
- Build Tier 0 labels by selecting a small subset of informative attributes, applying a linear model, adding noise, and calibrating the intercept to achieve ≈10% positive rate.
- Treat the oracle feature set as the raw attributes directly involved in the label, so ATTR and FE-Oracle use the same information.
- Provide a CLI entry (`python -m cli.run_experiment --tier tier0`) that generates the data, trains both arms with Optuna tuning, enforces a 5-minute timeout, and saves metrics under `artifacts/`.
- Ensure XGBoost uses all local CPU cores (`n_jobs=-1`).

## Near-term roadmap

1. Flesh out generators for higher tiers (oblique linear, pairwise compositions, etc.).
2. Add tier-aware feature constructors, evaluation tests, and richer reporting.
3. Introduce tier-specific property tests to guard data quality and training behavior.

This document will evolve as new tiers and capabilities are implemented.
