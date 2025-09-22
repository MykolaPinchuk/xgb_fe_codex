# Meta-Iteration 1 Findings & Hand-off Notes

## Status Update (post-denominator upgrade)
- Attribute generator continues to produce correlated Gaussian blocks with mixed scales, log-normal positive-only columns, and heavy-tailed distractors (see `src/dgp/attributes.py`).
- Ratio tiers now enforce mixed-scale denominators with CV guards (see `src/dgp/features_t2.py`, `features_t3.py`, `features_t4.py`). Tier 0–4 have been rerun with these updates (`artifacts/20250921_194128/…` through `artifacts/20250921_194714/…`), and diagnostics capture per-feature denominator CV.

### Result Summary (vs prior correlated backbone)

| Run                      | ATTR PR (old → new) | FE-Oracle PR (old → new) | ΔGap (pp) |
|--------------------------|---------------------|--------------------------|-----------|
| Tier 2 Ratio             | 0.465 → **0.629**   | 0.562 → **0.726**        | +0.13     |
| Tier 3 Ratio-of-Sums k=4 | 0.482 → **0.633**   | 0.651 → **0.780**        | −2.13     |
| Tier 4 Ratio-of-Sums k=6 | 0.612 → **0.551**   | 0.681 → **0.689**        | **+6.92** |

- Tier 4 ratio-of-sums regained difficulty (gap 13.8 pp, up from 6.9 pp) once denominators carried heterogeneous weights.
- Tier 3 ratios remain challenging (gap 14.8 pp) even though ATTR improved; denominator CVs typically ≥0.4 confirm the resampling guardrails are working.
- Tier 2 ratios stay stable but harder than before, with ATTR and oracle both benefitting from the wider weight pool.

## Immediate Next Steps
1. **Analyse refreshed results** — Tier 4 ratio-of-sums now shows a 13.8 pp PR-AUC gap (up from 6.9 pp), while Tier 3 ratio-of-sums remains challenging at 14.8 pp despite ATTR improving. Summarise these deltas against both the iid and early correlated baselines.
2. **Decide on follow-up experiments** such as relaxing denominator `abs()` handling or expanding XGBoost depth/search space.
3. **Incorporate diagnostics in summaries** (denominator CV, attribute variance) so future readers can quickly sanity-check backbone behaviour without digging into artifacts.

## Historical Observation (pre-upgrade)
- Tier 2+ results previously showed a flat ATTR vs FE-Oracle PR-AUC gap (~7–11 pp) despite increasing feature complexity because denominators were nearly constant under iid attributes.
- Correlation between raw numerators and ratio-of-sums features reached 0.79–0.94 on Tier 4 samples, confirming the near-linear effect.

## Open Questions
- Should we relax the blanket `abs` on denominators now that positive-only variance is meaningful? Consider staging this as a follow-on experiment.
- Do we want to widen the XGBoost search space (e.g., allow `max_depth=8`) after reviewing the new gap patterns?

## Current Repo Status (2025‑09‑21)
- Tier 0–4 generators and property tests pass (`pytest src/tests/test_tier3.py src/tests/test_tier4.py`).
- Latest Tier 4 runs: `artifacts/20250921_194643/tier4_ratioofsum_k6/` and `artifacts/20250921_194714/tier4_sumproduct_k6/` (mixed-scale denominators).
- Config defaults and documentation reflect the correlated/mixed-scale attribute generator plus denominator heterogeneity.
