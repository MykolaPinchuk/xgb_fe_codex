## High-Level Overview

### Goal

Test whether XGBoost trained on raw attributes (columns) can effectively "learn" the true features that drive the data-generating process (DGP).

Terminology

- `Attributes`: observed input columns (raw X).
- `Features`: the actual arguments entering the true DGP that produce labels (true z).

### Meta-Iteration 1: scope and constraints

- Binary classification with ~10% positive rate.
- No time dimension or hierarchies; all inputs numeric.
- Pure-tier runs only: each experiment uses a single subtier (feature group) as the sole source of true features in the DGP.

Each run has two training arms:

- ATTR — XGBoost trained on attributes only (`X`).
- FE-Oracle — XGBoost trained on the true features only (`z`) (the exact features used by the DGP).

We compare predictive performance (PR-AUC) and efficiency (trees to best_iteration) to evaluate how well trees recover different feature types from attributes alone.

### Why this design is fair

- We avoid mixing attributes and oracle features in the same arm because the research question is: "Can XGBoost on attributes emulate the DGP features?" Directly comparing ATTR vs FE-Oracle answers that.
- We use a single validation split (no CV) to keep runs fast; later iterations may add CV, stacked tiers, and a FE-Generic arm.

### Subtiers (feature groups) in Meta-Iteration 1

- **Tier 0 (baseline, not a DGP)**: attributes-only; used to sanity-check the training loop.
- **Tier 1 (oblique linear, k=2–4)**: features are linear combinations of a small subset of attributes (hyperplanes at angles to axes).
- **Tier 2 (pairwise compositional, k=2)**: products, ratios, order stats (min/max/|diff|).
- **Tier 3 (multi-var compositional, k=3–4)**: ratio-of-sums with unequal term counts and sum-product mixes like `(x1 + x2) * x3`.
- **Tier 4 (high-arity compositional, k=5–6)**: same templates as Tier 3 but with higher arity.

Tiers 5–7 (quadratic/distance, piecewise/hinge, logic-like) are out of scope for Meta-Iteration 1 and planned for later work.

### Metrics, hyperparameters, and speed knobs

- Primary metric: PR-AUC (suitable for ~10% minority rate).
- Secondary metrics: ROC-AUC and logloss (for sanity checks).
- Early stopping: 100 rounds monitored on PR-AUC.
- Optuna budget: 15 trials per arm.
- class imbalance handling: `scale_pos_weight = 2` (kept equal across arms for fairness and speed).
- Initial `max_depth`: 6 (fixed for Meta-Iteration 1; sweeps can be added later).

### Expected outcomes

- For oblique linear features, `FE-Oracle` should be modestly better and converge with fewer trees.
- For ratio/product and ratio-of-sums / sum-product features (especially at higher arity), `FE-Oracle` should show clearer gains and faster convergence.
- The pattern of gaps across tiers and arity will indicate where XGBoost on attributes struggles most.