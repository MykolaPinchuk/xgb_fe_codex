# XGB Feature Emulation Experiments

This project studies how well XGBoost trained on raw attributes can recover the latent features that generate binary labels. Meta-Iteration 1 focuses on synthetic data across "tiers" of feature complexity, with two model arms per run:

- **ATTR**: trains on raw attributes only.
- **FE-Oracle**: trains on the ground-truth features used by the data-generating process.

The first milestones cover Tier 0 (oracle equals selected raw attributes), Tier 1 (oblique linear combinations with configurable arity up to `k=6`), Tier 2 pairwise templates (products, ratios, abs-diffs, min/max), and Tier 3 ratio-of-sums / sum-product compositions. Future tiers will introduce higher-order stacks and additional feature families.

Detailed run notes are tracked under `docs/` (e.g., `docs/tier0.md`, `docs/tier1.md`, `docs/tier1_k6.md`, `docs/tier2_product.md`, `docs/tier2_ratio.md`, `docs/tier2_absdiff.md`, `docs/tier2_minmax.md`, `docs/tier3_ratioofsum.md`, `docs/tier3_sumproduct.md`, `docs/tier4_ratioofsum.md`, `docs/tier4_sumproduct.md`).

## Quick Metrics Summary

After executing experiments, summarize the latest run per tier/spec with:

```bash
python -m cli.summarize_runs --root artifacts --latest-only
```

See `IMPLEMENTATION.md` for the current scope and roadmap.

## Known Limitations

- Correlated, mixed-scale attributes are now in place (blocks with ρ≈0.4, positive-only log-normal transforms, heavy-tailed distractors) and ratio tiers enforce mixed-scale denominators. Tier 0–4 have been rerun (see latest `artifacts/20250921_1941xx–1947xx/…` directories), but deeper-tree sweeps and alternative objectives remain open questions.
