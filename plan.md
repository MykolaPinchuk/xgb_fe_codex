
# Detailed Implementation Plan & Specs

## 1) Project structure

repo/

- `README.md` — high-level overview (distilled from this doc).
- `IMPLEMENTATION.md` — detailed specs (lower sections of this doc distilled).
- `pyproject.toml` — dependencies (xgboost, scikit-learn, optuna, numpy, pandas).
- `src/`
  - `dgp/`
    - `attributes.py` — attribute generator (n, d, covariance, positive subset).
    - `features_t1.py` — oblique linear (k=2–4).
    - `features_t2.py` — pairwise (prod/ratio/order).
    - `features_t3.py` — k=3–4 ratio-of-sums & sum-product.
    - `features_t4.py` — k=5–6 ratio-of-sums & sum-product (higher arity).
    - `utils.py` — standardize, intercept solver, jitter, helpers.
  - `train/`
    - `run_arm.py` — single arm train/eval with Optuna.
    - `split.py` — stratified 80/20 split.
    - `metrics.py` — PR-AUC, ROC-AUC, logloss, trees-to-best.
  - `cli/`
    - `run_experiment.py` — orchestrates one subtier run: ATTR and FE-Oracle.
  - `tests/`
    - `test_generator.py` — property/sanity tests per subtier.
    - `test_training.py` — quick smoke tests for training loop.
- `artifacts/` — outputs per run (JSON, metrics.csv, params, logs).

## 2) Configuration (YAML + CLI)

Use a single YAML defaults file, overridable via CLI flags.

Example defaults for Meta-Iteration 1 (`config/defaults.yaml`):

```yaml
seed: 42
n_rows: 10000
n_attrs: 60
n_true_features:
  t1: 20
  t2: 20
  t3: 20
  t4: 15  # slightly lower at high arity for stability/speed
sigma_logit: 0.5
positive_rate: 0.10
val_frac: 0.20
xgb:
  tree_method: hist
  max_bin: 256
  max_depth: 6
  learning_rate_range: [0.01, 0.2]
  min_child_weight_range: [1.0, 20.0]
  subsample_range: [0.6, 1.0]
  colsample_bytree_range: [0.6, 1.0]
  reg_alpha_range: [1e-8, 10.0]
  reg_lambda_range: [1e-8, 10.0]
  scale_pos_weight: 2
  early_stopping_rounds: 100
optuna:
  n_trials: 15
```

CLI examples:

```bash
# Tier 1, k=3 (oblique linear features)
python -m cli.run_experiment --tier tier1 --k 3 --seed 123

# Tier 2, ratios (k=2)
python -m cli.run_experiment --tier tier2 --spec ratio --seed 123

# Tier 3, k=4, sum-product
python -m cli.run_experiment --tier tier3 --k 4 --spec sumproduct --seed 123

# Tier 4, k=6, ratio-of-sums (high arity)
python -m cli.run_experiment --tier tier4 --k 6 --spec ratioofsum --seed 123
```

Each run produces two arms (ATTR, FE-Oracle) and writes metrics + params to `artifacts/<date_time>/<tier>_<spec>_k<k>/`.

## 3) Data generation (attributes)

- Total attributes: `d = 60`.
- 24 informative-eligible attributes (pool used by true features):
  - (Meta-Iteration 1 implementation) iid Gaussian draws with half of the informative set exponentiated to enforce positivity.
  - (Deferred to Meta-Iteration 2) correlated Gaussian blocks (e.g., block size 6 with intra-block ρ≈0.4) and mixed scales.
- 36 distractor attributes: independent Gaussian / mild heavy-tail mix.

Return `X` as a `pandas.DataFrame` with columns `x0..x59` and a metadata structure indicating which indices are positive-only.

## 4) Feature construction per subtier (true features `z`)

Shared rules

- Build `N` features (`N=20` for T1–T3, `N=15` for T4 by default).
- Ensure coverage: every informative-eligible attribute appears in ≥1 feature; approximate uniform usage.
- Standardize each `z` to mean 0, std 1.
- Use weights from `{±1, ±0.5, ±2}` unless specified.
- Use `ε = 1e-3` for denominators; apply `abs()` in denominators when attributes can be negative.

Tier templates

- **Tier 1 (oblique linear, k=2–4)**
  - `z = sum_j w_j * x_j` with `|S| = k` (2, 3, or 4).
- **Tier 2 (pairwise, k=2)**
  - `prod(i,j) = x_i * x_j`
  - `ratio(i,j) = x_i / (|x_j| + ε)` (prefer `j` from positive subset or use `abs`).
  - `absdiff(i,j) = |x_i - x_j|`
  - `min(i,j), max(i,j)`
- **Tier 3 (k=3–4)**
  - Unequal-term ratios: `(sum_{i∈S} a_i x_i) / (sum_{j∈T} b_j |x_j| + ε)` with `|S| ∈ {1,2}` and `|T| ∈ {2,3,4}`.
  - Sum-product mixes: `(sum_{i∈S} a_i x_i) * x_k` or `(sum_{i∈S} a_i x_i) * (sum_{j∈T} c_j x_j)` (total k=3–4).
- **Tier 4 (k=5–6)**
  - Same templates as Tier 3 but with higher arity: `k ∈ {5,6}` and default `N=15`.

## 5) DGP (logit) and labels

1. Construct features `z1..zN` per subtier.
2. Sample coefficients `β_i ~ Uniform(0.5, 1.5)` with random sign.
3. Compute logit: `f = sum_i β_i * z_i`.
4. Add jitter `η ~ Normal(0, σ^2)`, default `σ = 0.5`.
5. Find intercept `b` by bisection so that `P(y=1) ≈ 0.10` on the full sample.
6. Generate labels: `y ~ Bernoulli(sigmoid(f + b + η))`.

Return `(X, z_df, y)`; we feed `X` to the ATTR arm and `z_df` to the FE-Oracle arm.

## 6) Train/eval loop per arm

- Split: stratified 80/20 (fixed seed).
- Impute: median.
- Model: `XGBClassifier` with hyperparameter ranges from config.
- Objective/metric: monitor PR-AUC on the validation split; use `early_stopping_rounds = 100`.
- Imbalance handling: `scale_pos_weight = 2`.
- Search: Optuna with `n_trials = 15` maximizing PR-AUC.

Record and save per arm:

- `pr_auc`, `roc_auc`, `logloss` on validation.
- `best_iteration` (trees to best), `fit_seconds`.
- Best hyperparameters.

Save outputs as `metrics_ATTR.json`, `metrics_FEORACLE.json`, and parameter files under the run's artifacts directory.

## 7) Sanity checks and quick validations

- Oracle logistic (sanity): `LogisticRegression` on `z_df` should reach strong PR-AUC (comparable to FE-Oracle XGB).
- Class rate: empirical positive rate in `[0.095, 0.105]`.
- Denominators: log min/median denominators where applicable; assert `min ≥ ε` after `abs`/positive handling.

## 8) Test suite (to run after each subtier is implemented)

A. Generator property tests (fast, `n=2000`)

- Prevalence test: empirical `mean(y)` within 1% of `0.10`.
- Coverage test: every informative-eligible attribute appears in at least one feature.
- Numerical safety: denominators `min ≥ ε`, no NaNs/inf in `z`.
- Standardization test: each `z` has `|mean| < 1e-6`, `std ≈ 1`.

B. Training smoke tests (fast, `n=5000`, `depth=4`, `trials=5`)

- Loop runs: both arms train without error and save artifacts.
- Basic performance floor: `FE-Oracle PR-AUC ≥ 0.60` (T1/T2) or `≥ 0.65` (T3/T4) on the small sample.
- Non-negativity check: `FE-Oracle PR-AUC ≥ ATTR PR-AUC − 0.01` (tolerate small noise).

C. Efficiency check (default `n=10k`)

- Trees to best: `FE-Oracle best_iteration ≤ ATTR best_iteration` in the majority of runs (log the fraction; no hard assert).

D. Reproducibility

- With fixed seed and config, metric values stable within tolerance (e.g., ΔPR-AUC drift < 0.01).

## 9) Artifacts & logging

Directory layout: `artifacts/<YYYYmmdd_HHMMSS>/<tier>_<spec>_k<k>/`

- `config_used.json`
- `metrics_ATTR.json`, `metrics_FEORACLE.json`
- `best_params_ATTR.json`, `best_params_FEORACLE.json`
- `timings.json` (fit seconds per arm)
- `denominator_stats.json` (where applicable)
- `seed_info.txt`
- Optional: save XGBoost boosters (`.json` or `.ubj`).

## 10) Runbook (Meta-Iteration 1)

1. Implement Tier 1 → run tests A/B/C on T1; if green, run the full default (`n=10k`) experiment and commit artifacts.
2. Implement Tier 2 → run tests; then full run and commit artifacts.
3. Implement Tier 3 (k=3 and k=4 variants) → tests, then full runs.
4. Implement Tier 4 (k=5 and k=6 variants) → tests, then full runs.
5. Add a simple summary script to aggregate `metrics_*.json` and print a comparison table (ATTR vs FE-Oracle by subtier).
6. **Meta-Iteration 2 kickoff:** upgrade the attribute generator (correlated blocks, mixed scales) before rerunning Tier 2–4. See `docs/hand_off_notes.md` for context and open questions.

## 11) Notes & small gotchas

- `scale_pos_weight=2` is lighter than the `1/p - 1` heuristic (≈9 for 10% positives); acceptable for a speed-focused pass since PR-AUC is the monitored metric.
- Early stopping on PR-AUC is crucial; don't switch to ROC-AUC for early stopping.
- For k=5–6 features (Tier 4), keep `N=15` default. If training looks noisy, consider `σ=0.4` or narrower `β` ranges (config-gated) to stabilize.
- We want toietratie rapidly. use all threads for model training. Makse sure that no training run (including hpo and evth else) exceeds 5 minutes. use timeouts if needed to enfornce it.
- do not overnegineer.

### After Meta-Iteration 1 (preview)

- Add stacked-tier runs (mix current and lower tiers).
- Introduce a `FE-Generic` arm (a blind composite feature kit) and compare ATTR vs FE-Generic vs FE-Oracle.
- Add CV and depth sweeps to study capacity vs engineered features.
- Extend to later tiers (quadratic/distance, hinges, logic-like).
