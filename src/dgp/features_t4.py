from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import sample_logistic_labels, standardize_columns


DEFAULT_NUMERATOR_POOL = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
DEFAULT_DENOMINATOR_POOL = np.array([0.5, 1.0, 2.0])
DEFAULT_SUM_POOL = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])


@dataclass
class Tier4Config:
    positive_rate: float
    sigma_logit: float
    k: int
    n_features: int = 15
    spec: str = "ratioofsum"
    epsilon: float = 1e-3
    numerator_weight_pool: Sequence[float] = tuple(DEFAULT_NUMERATOR_POOL)
    denominator_weight_pool: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0)
    sum_weight_pool: Sequence[float] = tuple(DEFAULT_SUM_POOL)
    positive_only_indices: List[int] | None = None
    min_denominator_cv: float = 0.35
    max_denominator_resamples: int = 5


@dataclass
class Tier4Outputs:
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


def _draw_weights(rng: np.random.Generator, pool: Sequence[float], size: int) -> np.ndarray:
    weights = rng.choice(pool, size=size, replace=True)
    return weights.astype(float)


def _choose_ratio_sizes(k: int, rng: np.random.Generator) -> Tuple[int, int]:
    # Favor numerator sizes 2–3, denominator takes the remainder (>=2).
    numerator_choices = [size for size in (2, 3) if 2 <= size <= k - 2]
    numerator = rng.choice(numerator_choices)
    denominator = k - numerator
    return int(numerator), int(denominator)


def _choose_sumproduct_structure(k: int, rng: np.random.Generator) -> Tuple[str, Dict[str, int]]:
    if k == 5:
        options = [
            ("sum_times_single", {"left": 4, "right": 1}),
            ("sum_times_sum", {"left": 3, "right": 2}),
        ]
    elif k == 6:
        options = [
            ("sum_times_single", {"left": 5, "right": 1}),
            ("sum_times_sum", {"left": 3, "right": 3}),
            ("sum_times_sum", {"left": 4, "right": 2}),
        ]
    else:
        raise ValueError(f"Tier4 sum-product expects k in {{5, 6}}, received {k}")
    return options[int(rng.integers(len(options)))]


def _generate_ratio_of_sums(
    X: pd.DataFrame,
    informative_indices: List[int],
    config: Tier4Config,
    rng: np.random.Generator,
) -> Tier4Outputs:
    if config.k not in {5, 6}:
        raise ValueError(f"Tier4 ratio-of-sums expects k in {{5,6}}, received {config.k}")
    if len(informative_indices) < config.k:
        raise ValueError("Number of informative attributes must be >= k for Tier4 ratio-of-sums")

    n_rows = X.shape[0]
    feature_defs: List[Dict[str, object]] = []
    attribute_usage: Dict[str, int] = {X.columns[idx]: 0 for idx in informative_indices}
    raw_features = np.zeros((n_rows, config.n_features), dtype=float)
    positive_set = set(config.positive_only_indices or [])

    coverage_queue = informative_indices.copy()
    rng.shuffle(coverage_queue)

    for feat_idx in range(config.n_features):
        numerator_size, denominator_size = _choose_ratio_sizes(config.k, rng)
        chosen: List[int] = []

        def select_group(required: int) -> List[int]:
            group: List[int] = []
            while len(group) < required:
                candidate = _pop_unique(coverage_queue, group + chosen)
                if candidate is None:
                    candidate = int(rng.choice(informative_indices))
                if candidate not in group and candidate not in chosen:
                    group.append(candidate)
            return group

        numerator_indices = select_group(numerator_size)
        chosen.extend(numerator_indices)
        denominator_indices = select_group(denominator_size)
        if positive_set and denominator_size >= 2:
            if not any(idx in positive_set for idx in denominator_indices):
                replacement_candidates = [idx for idx in informative_indices if idx in positive_set and idx not in numerator_indices and idx not in denominator_indices]
                if replacement_candidates:
                    replace_idx = rng.integers(0, len(denominator_indices))
                    denominator_indices[replace_idx] = int(rng.choice(replacement_candidates))
        non_positive_candidates = [idx for idx in informative_indices if idx not in positive_set and idx not in numerator_indices and idx not in denominator_indices]
        if non_positive_candidates and denominator_size >= 2:
            if not any(idx not in positive_set for idx in denominator_indices):
                replace_idx = rng.integers(0, len(denominator_indices))
                denominator_indices[replace_idx] = int(rng.choice(non_positive_candidates))
        chosen.extend(denominator_indices)

        numerator_cols = [X.columns[idx] for idx in numerator_indices]
        denominator_cols = [X.columns[idx] for idx in denominator_indices]

        numerator_weights = _draw_weights(rng, config.numerator_weight_pool, size=len(numerator_indices))

        denom_resample_attempts = 0
        denominator_weights = None
        denom_vals = None
        while denom_resample_attempts < config.max_denominator_resamples:
            denominator_weights = _draw_weights(rng, config.denominator_weight_pool, size=len(denominator_indices))
            scale_multipliers = rng.lognormal(mean=0.0, sigma=0.6, size=len(denominator_indices))
            denominator_weights = denominator_weights * scale_multipliers

            numerator_vals = (X.iloc[:, numerator_indices].to_numpy() * numerator_weights).sum(axis=1)
            denominator_inputs = np.abs(X.iloc[:, denominator_indices].to_numpy())
            denom_vals = (denominator_inputs * denominator_weights).sum(axis=1) + config.epsilon
            mean = float(denom_vals.mean())
            std = float(denom_vals.std(ddof=0))
            cv = std / mean if mean > 0 else float("inf")
            if cv >= config.min_denominator_cv or denom_resample_attempts == config.max_denominator_resamples - 1:
                break
            denom_resample_attempts += 1

        for col in numerator_cols:
            attribute_usage[col] = attribute_usage.get(col, 0) + 1
        for col in denominator_cols:
            attribute_usage[col] = attribute_usage.get(col, 0) + 1

        raw_features[:, feat_idx] = numerator_vals / denom_vals

        feature_defs.append(
            {
                "type": "ratioofsum",
                "numerator": {col: float(w) for col, w in zip(numerator_cols, numerator_weights)},
                "denominator": {col: float(w) for col, w in zip(denominator_cols, denominator_weights)},
                "denominator_cv": float(
                    float(np.std(denom_vals, ddof=0)) / float(np.mean(denom_vals))
                    if np.mean(denom_vals) > 0
                    else float("inf")
                ),
            }
        )

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

    return Tier4Outputs(
        features=features_df,
        labels=labels,
        feature_defs=feature_defs,
        betas=betas.tolist(),
        intercept=float(intercept),
        attribute_usage=attribute_usage,
    )


def _generate_sum_product(
    X: pd.DataFrame,
    informative_indices: List[int],
    config: Tier4Config,
    rng: np.random.Generator,
) -> Tier4Outputs:
    if config.k not in {5, 6}:
        raise ValueError(f"Tier4 sum-product expects k in {{5,6}}, received {config.k}")
    if len(informative_indices) < config.k:
        raise ValueError("Number of informative attributes must be >= k for Tier4 sum-product")

    n_rows = X.shape[0]
    feature_defs: List[Dict[str, object]] = []
    attribute_usage: Dict[str, int] = {X.columns[idx]: 0 for idx in informative_indices}
    raw_features = np.zeros((n_rows, config.n_features), dtype=float)

    coverage_queue = informative_indices.copy()
    rng.shuffle(coverage_queue)

    for feat_idx in range(config.n_features):
        structure, sizes = _choose_sumproduct_structure(config.k, rng)
        chosen: List[int] = []

        def select_group(required: int) -> List[int]:
            group: List[int] = []
            while len(group) < required:
                candidate = _pop_unique(coverage_queue, group + chosen)
                if candidate is None:
                    candidate = int(rng.choice(informative_indices))
                if candidate not in group and candidate not in chosen:
                    group.append(candidate)
            return group

        left_indices = select_group(sizes["left"])
        chosen.extend(left_indices)
        right_indices = select_group(sizes["right"])
        chosen.extend(right_indices)

        left_cols = [X.columns[idx] for idx in left_indices]
        right_cols = [X.columns[idx] for idx in right_indices]

        left_weights = _draw_weights(rng, config.sum_weight_pool, size=len(left_indices))

        if structure == "sum_times_single":
            single_idx = right_indices[0]
            single_col = right_cols[0]
            single_weight = _draw_weights(rng, config.sum_weight_pool, size=1)[0]

            for col in left_cols + [single_col]:
                attribute_usage[col] = attribute_usage.get(col, 0) + 1

            left_vals = (X.iloc[:, left_indices].to_numpy() * left_weights).sum(axis=1)
            right_vals = single_weight * X.iloc[:, single_idx].to_numpy()

            feature_defs.append(
                {
                    "type": "sumproduct",
                    "structure": structure,
                    "left": {col: float(w) for col, w in zip(left_cols, left_weights)},
                    "right": {single_col: float(single_weight)},
                }
            )
        else:
            right_weights = _draw_weights(rng, config.sum_weight_pool, size=len(right_indices))

            for col in left_cols + right_cols:
                attribute_usage[col] = attribute_usage.get(col, 0) + 1

            left_vals = (X.iloc[:, left_indices].to_numpy() * left_weights).sum(axis=1)
            right_vals = (X.iloc[:, right_indices].to_numpy() * right_weights).sum(axis=1)

            feature_defs.append(
                {
                    "type": "sumproduct",
                    "structure": structure,
                    "left": {col: float(w) for col, w in zip(left_cols, left_weights)},
                    "right": {col: float(w) for col, w in zip(right_cols, right_weights)},
                }
            )

        raw_features[:, feat_idx] = left_vals * right_vals

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

    return Tier4Outputs(
        features=features_df,
        labels=labels,
        feature_defs=feature_defs,
        betas=betas.tolist(),
        intercept=float(intercept),
        attribute_usage=attribute_usage,
    )


def generate_tier4(
    X: pd.DataFrame,
    informative_indices: List[int],
    config: Tier4Config,
    rng: np.random.Generator,
) -> Tier4Outputs:
    spec = config.spec.lower()
    if spec == "ratioofsum":
        return _generate_ratio_of_sums(X, informative_indices, config, rng)
    if spec == "sumproduct":
        return _generate_sum_product(X, informative_indices, config, rng)
    raise ValueError(f"Unsupported Tier4 spec: {config.spec}")
