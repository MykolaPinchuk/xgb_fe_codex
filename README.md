# XGB Feature Emulation Experiments

This project studies how well XGBoost trained on raw attributes can recover the latent features that generate binary labels. Meta-Iteration 1 focuses on synthetic data across "tiers" of feature complexity, with two model arms per run:

- **ATTR**: trains on raw attributes only.
- **FE-Oracle**: trains on the ground-truth features used by the data-generating process.

The first milestones cover Tier 0 (oracle equals selected raw attributes), Tier 1 (oblique linear combinations with configurable arity up to `k=6`), and Tier 2 pairwise templates (products, ratios, abs-diffs, min/max). Future tiers will introduce higher-order compositions and stacked feature families.

Detailed run notes are tracked under `docs/` (e.g., `docs/tier0.md`, `docs/tier1.md`, `docs/tier1_k6.md`, `docs/tier2_product.md`, `docs/tier2_ratio.md`, `docs/tier2_absdiff.md`, `docs/tier2_minmax.md`).

See `IMPLEMENTATION.md` for the current scope and roadmap.
