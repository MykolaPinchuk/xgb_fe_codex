from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .utils import sample_logistic_labels, standardize_columns


@dataclass
class Tier2Config:
    positive_rate: float
    sigma_logit: float
    n_features: int
    spec: str = "product"
    positive_only_indices: List[int] | None = None
    denominator_weight_pool: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass
class Tier2Outputs:
    features: pd.DataFrame
    labels: pd.Series
    feature_defs: List[Dict[str, object]]
    betas: List[float]
    intercept: float
    attribute_usage: Dict[str, int]


def _pop_unique(queue: List[int], chosen: List[int]) -> int | None:
    while queue:
        candidate = queue.pop()
        if candidate not in chosen:
            return candidate
    return None


def _generate_product_features(
    X: pd.DataFrame,
    informative_indices: List[int],
    config: Tier2Config,
    rng: np.random.Generator,
) -> Tier2Outputs:
    if len(informative_indices) < 2:
        raise ValueError("Tier2 product spec requires at least two informative attributes")

    n_rows = X.shape[0]
    feature_defs: List[Dict[str, object]] = []
    attribute_usage: Dict[str, int] = {X.columns[idx]: 0 for idx in informative_indices}
    raw_features = np.zeros((n_rows, config.n_features), dtype=float)

    coverage_queue = informative_indices.copy()
    rng.shuffle(coverage_queue)

    for feat_idx in range(config.n_features):
        chosen: List[int] = []

        first = _pop_unique(coverage_queue, chosen)
        if first is None:
            first = int(rng.choice(informative_indices))
        chosen.append(first)

        second = _pop_unique(coverage_queue, chosen)
        if second is None:
            choices = [idx for idx in informative_indices if idx != first]
            second = int(rng.choice(choices))
        chosen.append(second)

        idx_a, idx_b = sorted(chosen)
        col_a, col_b = X.columns[idx_a], X.columns[idx_b]

        vals = X.iloc[:, idx_a].to_numpy() * X.iloc[:, idx_b].to_numpy()
        raw_features[:, feat_idx] = vals

        feature_defs.append({"type": "product", "cols": [col_a, col_b]})
        attribute_usage[col_a] = attribute_usage.get(col_a, 0) + 1
        attribute_usage[col_b] = attribute_usage.get(col_b, 0) + 1

    standardized = standardize_columns(raw_features)
    feature_cols = [f"z{i}" for i in range(config.n_features)]
    features_df = pd.DataFrame(standardized, columns=feature_cols, index=X.index)

    betas = rng.uniform(0.5, 1.5, size=config.n_features)
    betas *= rng.choice([-1.0, 1.0], size=config.n_features)

    logits = standardized @ betas
    labels_array, intercept, _ = sample_logistic_labels(
        logits=logits,
        positive_rate=config.positive_rate,
        sigma=config.sigma_logit,
        rng=rng,
    )

    labels = pd.Series(labels_array, name="y", index=X.index)

    return Tier2Outputs(
        features=features_df,
        labels=labels,
        feature_defs=feature_defs,
        betas=betas.tolist(),
        intercept=float(intercept),
        attribute_usage=attribute_usage,
    )


def _generate_absdiff_features(
    X: pd.DataFrame,
    informative_indices: List[int],
    config: Tier2Config,
    rng: np.random.Generator,
) -> Tier2Outputs:
    if len(informative_indices) < 2:
        raise ValueError("Tier2 absdiff spec requires at least two informative attributes")

    n_rows = X.shape[0]
    feature_defs: List[Dict[str, object]] = []
    attribute_usage: Dict[str, int] = {X.columns[idx]: 0 for idx in informative_indices}
    raw_features = np.zeros((n_rows, config.n_features), dtype=float)

    coverage_queue = informative_indices.copy()
    rng.shuffle(coverage_queue)

    for feat_idx in range(config.n_features):
        chosen: List[int] = []

        first = _pop_unique(coverage_queue, chosen)
        if first is None:
            first = int(rng.choice(informative_indices))
        chosen.append(first)

        second = _pop_unique(coverage_queue, chosen)
        if second is None:
            candidates = [idx for idx in informative_indices if idx != first]
            second = int(rng.choice(candidates))
        chosen.append(second)

        idx_a, idx_b = sorted(chosen)
        col_a, col_b = X.columns[idx_a], X.columns[idx_b]

        vals = np.abs(X.iloc[:, idx_a].to_numpy() - X.iloc[:, idx_b].to_numpy())
        raw_features[:, feat_idx] = vals

        feature_defs.append({"type": "absdiff", "cols": [col_a, col_b]})
        attribute_usage[col_a] = attribute_usage.get(col_a, 0) + 1
        attribute_usage[col_b] = attribute_usage.get(col_b, 0) + 1

    standardized = standardize_columns(raw_features)
    feature_cols = [f"z{i}" for i in range(config.n_features)]
    features_df = pd.DataFrame(standardized, columns=feature_cols, index=X.index)

    betas = rng.uniform(0.5, 1.5, size=config.n_features)
    betas *= rng.choice([-1.0, 1.0], size=config.n_features)

    logits = standardized @ betas
    labels_array, intercept, _ = sample_logistic_labels(
        logits=logits,
        positive_rate=config.positive_rate,
        sigma=config.sigma_logit,
        rng=rng,
    )

    labels = pd.Series(labels_array, name="y", index=X.index)

    return Tier2Outputs(
        features=features_df,
        labels=labels,
        feature_defs=feature_defs,
        betas=betas.tolist(),
        intercept=float(intercept),
        attribute_usage=attribute_usage,
    )


def _generate_extrema_features(
    X: pd.DataFrame,
    informative_indices: List[int],
    config: Tier2Config,
    rng: np.random.Generator,
) -> Tier2Outputs:
    if len(informative_indices) < 2:
        raise ValueError("Tier2 minmax spec requires at least two informative attributes")

    n_rows = X.shape[0]
    feature_defs: List[Dict[str, object]] = []
    attribute_usage: Dict[str, int] = {X.columns[idx]: 0 for idx in informative_indices}
    raw_features = np.zeros((n_rows, config.n_features), dtype=float)

    coverage_queue = informative_indices.copy()
    rng.shuffle(coverage_queue)

    for feat_idx in range(config.n_features):
        chosen: List[int] = []
        first = _pop_unique(coverage_queue, chosen)
        if first is None:
            first = int(rng.choice(informative_indices))
        chosen.append(first)

        second = _pop_unique(coverage_queue, chosen)
        if second is None:
            second = int(rng.choice([idx for idx in informative_indices if idx != first]))
        chosen.append(second)

        idx_a, idx_b = sorted(chosen)

        col_a, col_b = X.columns[idx_a], X.columns[idx_b]
        values_a = X.iloc[:, idx_a].to_numpy()
        values_b = X.iloc[:, idx_b].to_numpy()

        if feat_idx % 2 == 0:
            vals = np.minimum(values_a, values_b)
            feature_type = "min"
        else:
            vals = np.maximum(values_a, values_b)
            feature_type = "max"

        raw_features[:, feat_idx] = vals
        feature_defs.append({"type": feature_type, "cols": [col_a, col_b]})

        attribute_usage[col_a] = attribute_usage.get(col_a, 0) + 1
        attribute_usage[col_b] = attribute_usage.get(col_b, 0) + 1

    standardized = standardize_columns(raw_features)
    feature_cols = [f"z{i}" for i in range(config.n_features)]
    features_df = pd.DataFrame(standardized, columns=feature_cols, index=X.index)

    betas = rng.uniform(0.5, 1.5, size=config.n_features)
    betas *= rng.choice([-1.0, 1.0], size=config.n_features)

    logits = standardized @ betas
    labels_array, intercept, _ = sample_logistic_labels(
        logits=logits,
        positive_rate=config.positive_rate,
        sigma=config.sigma_logit,
        rng=rng,
    )

    labels = pd.Series(labels_array, name="y", index=X.index)

    return Tier2Outputs(
        features=features_df,
        labels=labels,
        feature_defs=feature_defs,
        betas=betas.tolist(),
        intercept=float(intercept),
        attribute_usage=attribute_usage,
    )


def _generate_ratio_features(
    X: pd.DataFrame,
    informative_indices: List[int],
    config: Tier2Config,
    rng: np.random.Generator,
) -> Tier2Outputs:
    if len(informative_indices) < 2:
        raise ValueError("Tier2 ratio spec requires at least two informative attributes")

    n_rows = X.shape[0]
    eps = 1e-3
    feature_defs: List[Dict[str, object]] = []
    attribute_usage: Dict[str, int] = {X.columns[idx]: 0 for idx in informative_indices}
    raw_features = np.zeros((n_rows, config.n_features), dtype=float)
    variances = X.var(axis=0, ddof=0).to_numpy()

    coverage_queue = informative_indices.copy()
    rng.shuffle(coverage_queue)

    positive_set = set(config.positive_only_indices or [])

    for feat_idx in range(config.n_features):
        chosen: List[int] = []

        numerator = _pop_unique(coverage_queue, chosen)
        if numerator is None:
            numerator = int(rng.choice(informative_indices))
        chosen.append(numerator)

        denom_candidate = _pop_unique(coverage_queue, chosen)
        positive_set = set(config.positive_only_indices or [])
        if denom_candidate is None:
            denom_pool = [idx for idx in informative_indices if idx != numerator]
            if positive_set:
                pos_pool = [idx for idx in denom_pool if idx in positive_set]
                if pos_pool:
                    probs = variances[pos_pool]
                    probs = probs / probs.sum() if probs.sum() > 0 else None
                    denom_candidate = int(
                        rng.choice(pos_pool, p=probs if probs is not None else None)
                    )
            if denom_candidate is None:
                denom_pool = [idx for idx in denom_pool if idx not in positive_set] or denom_pool
                probs = variances[denom_pool]
                probs = probs / probs.sum() if probs.sum() > 0 else None
                denom_candidate = int(
                    rng.choice(denom_pool, p=probs if probs is not None else None)
                )
        chosen.append(denom_candidate)

        num_idx = numerator
        den_idx = denom_candidate
        num_col = X.columns[num_idx]
        den_col = X.columns[den_idx]

        numerator_vals = X.iloc[:, num_idx].to_numpy()
        denominator_vals = X.iloc[:, den_idx].to_numpy()
        if den_idx not in positive_set:
            denominator_vals = np.abs(denominator_vals)
        weight = float(rng.choice(config.denominator_weight_pool))
        denominator_vals = denominator_vals * weight + eps

        vals = numerator_vals / denominator_vals
        raw_features[:, feat_idx] = vals

        feature_defs.append(
            {
                "type": "ratio",
                "numerator": num_col,
                "denominator": den_col,
                "denominator_weight": weight,
            }
        )
        attribute_usage[num_col] = attribute_usage.get(num_col, 0) + 1
        attribute_usage[den_col] = attribute_usage.get(den_col, 0) + 1

    standardized = standardize_columns(raw_features)
    feature_cols = [f"z{i}" for i in range(config.n_features)]
    features_df = pd.DataFrame(standardized, columns=feature_cols, index=X.index)

    betas = rng.uniform(0.5, 1.5, size=config.n_features)
    betas *= rng.choice([-1.0, 1.0], size=config.n_features)

    logits = standardized @ betas
    labels_array, intercept, _ = sample_logistic_labels(
        logits=logits,
        positive_rate=config.positive_rate,
        sigma=config.sigma_logit,
        rng=rng,
    )

    labels = pd.Series(labels_array, name="y", index=X.index)

    return Tier2Outputs(
        features=features_df,
        labels=labels,
        feature_defs=feature_defs,
        betas=betas.tolist(),
        intercept=float(intercept),
        attribute_usage=attribute_usage,
    )


def generate_tier2(
    X: pd.DataFrame,
    informative_indices: List[int],
    config: Tier2Config,
    rng: np.random.Generator,
) -> Tier2Outputs:
    spec = config.spec.lower()
    if spec == "product":
        return _generate_product_features(X, informative_indices, config, rng)
    if spec == "ratio":
        return _generate_ratio_features(X, informative_indices, config, rng)
    if spec == "absdiff":
        return _generate_absdiff_features(X, informative_indices, config, rng)
    if spec == "minmax":
        return _generate_extrema_features(X, informative_indices, config, rng)
    raise ValueError(f"Unsupported Tier2 spec: {config.spec}")
