# XGB Feature Emulation Experiments

This project studies how well XGBoost trained on raw attributes can recover the latent features that generate binary labels. Meta-Iteration 1 focuses on synthetic data across "tiers" of feature complexity, with two model arms per run:

- **ATTR**: trains on raw attributes only.
- **FE-Oracle**: trains on the ground-truth features used by the data-generating process.

The initial milestone implements Tier 0, a sanity check where oracle features match a handful of raw attributes. Future tiers introduce increasingly compositional features (oblique linear, pairwise compositions, ratio-of-sums, etc.).

See `IMPLEMENTATION.md` for the current scope and roadmap.
