# Meta-Iteration 1 Findings & Hand-off Notes

## Key Observation
- Tier 2+ results show a nearly flat ATTR vs FE-Oracle PR-AUC gap (~7–11 pp) despite increasing feature complexity.
- Investigation of Tier 4 ratio-of-sums features (`src/dgp/features_t4.py`) shows denominators are nearly constant because the attribute generator (`src/dgp/attributes.py`) emits iid normals with half-lognormal positives. Ratios therefore reduce to rescaled numerators, making higher-tier features almost linear.

## Evidence
- Correlation between the raw numerator and the fully constructed ratio-of-sums feature reaches 0.79–0.94 on sampled Tier 4 features, confirming the near-linear effect.
- Quick re-runs with small Optuna budgets (`python - <<'PY' ...`) reveal FE-Oracle still outperforms ATTR, but the extra complexity offers only marginal gains because ATTR sees an easy-to-approximate signal.

## Recommended Next Steps (Meta-Iteration 2)
1. **Upgrade attribute generator**
   - Introduce correlated Gaussian blocks (ρ≈0.3–0.5), mixed scales, and heavier tails as described in `plan.md`.
   - Ensure positive-only attributes still exist but vary enough that denominators fluctuate materially.
2. **Revisit ratio templates**
   - Consider removing the blanket `abs` on denominators when the positive-only pool is used; instead, draw denominators preferentially from the positive subset and allow signed denominators elsewhere.
   - Track denominator statistics per run to validate spread (min/median/std) and log them under `artifacts/.../diagnostics.json`.
3. **Rerun Tier 2–4 experiments** with the richer attribute backbone. Expect larger PR-AUC gaps and potentially higher tree counts; adjust Optuna budgets if necessary.
4. **Update documentation** once the new generator is in place: add a "Known Limitations" note describing the previous iid-attribute behaviour and summarize post-upgrade results.

## Open Questions for the Next Agent
- Should we tune XGBoost depth (e.g., allow `max_depth=8`) to check whether deeper trees shrink the gaps on richer data?
- Do we want to log additional diagnostics (feature correlations, variance ratios) to track how challenging each tier becomes after the upgrade?

## Current Repo Status (2025‑09‑21)
- Tier 0–4 generators and property tests pass (`pytest src/tests/test_tier3.py src/tests/test_tier4.py`).
- Latest Tier 4 runs: `artifacts/20250921_183528/tier4_ratioofsum_k6/` and `artifacts/20250921_183727/tier4_sumproduct_k6/`.
- Config defaults remain tuned for the simple attribute generator; expect to revisit ranges once the new backbone lands.
