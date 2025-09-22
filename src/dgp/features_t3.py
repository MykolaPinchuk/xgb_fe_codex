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
class Tier3Config:
    positive_rate: float
    sigma_logit: float
    k: int
    n_features: int = 20
    spec: str = "ratioofsum"
    epsilon: float = 1e-3
    numerator_weight_pool: Sequence[float] = tuple(DEFAULT_NUMERATOR_POOL)
    denominator_weight_pool: Sequence[float] = tuple(DEFAULT_DENOMINATOR_POOL)
    sum_weight_pool: Sequence[float] = tuple(DEFAULT_SUM_POOL)


@dataclass
class Tier3Outputs:
    features: pd.DataFrame
    labels: pd.Series
    feature_defs: List[Dict[str, object]]
    betas: List[float]
    intercept: float
    attribute_usage: Dict[str, int]


def _choose_group_sizes(k: int, rng: np.random.Generator) -> Tuple[int, int]:
    if k == 3:
        return 1, 2
    if k == 4:
        # Alternate between (1,3) and (2,2) to diversify feature structures.
        return (1, 3) if rng.uniform() < 0.5 else (2, 2)
    raise ValueError(f"Tier3 ratio-of-sums expects k in {{3, 4}}, received {k}")


def _pop_unique(queue: List[int], chosen: List[int]) -> int | None:
    while queue:
        candidate = queue.pop()
        if candidate not in chosen:
            return candidate
    return None


def _draw_weights(rng: np.random.Generator, pool: Sequence[float], size: int) -> np.ndarray:
    weights = rng.choice(pool, size=size, replace=True)
    return weights.astype(float)


def _choose_sum_product_structure(k: int, rng: np.random.Generator) -> Tuple[str, Dict[str, int]]:
    if k == 3:
        # (sum of 2) * (single)
        return "sum_times_single", {"left": 2, "right": 1}
    if k == 4:
        if rng.uniform() < 0.5:
            # (sum of 3) * (single)
            return "sum_times_single", {"left": 3, "right": 1}
        # (sum of 2) * (sum of 2)
        return "sum_times_sum", {"left": 2, "right": 2}
    raise ValueError(f"Tier3 sum-product expects k in {{3, 4}}, received {k}")


def _generate_ratio_of_sums(
    X: pd.DataFrame,
    informative_indices: List[int],
    config: Tier3Config,
    rng: np.random.Generator,
) -> Tier3Outputs:
    if config.k < 3 or config.k > 4:
        raise ValueError(f"Tier3 ratio-of-sums expects k in [3,4], received {config.k}")
    if len(informative_indices) < config.k:
        raise ValueError("Number of informative attributes must be >= k for Tier3 ratio-of-sums")

    n_rows = X.shape[0]
    feature_defs: List[Dict[str, object]] = []
    attribute_usage: Dict[str, int] = {X.columns[idx]: 0 for idx in informative_indices}
    raw_features = np.zeros((n_rows, config.n_features), dtype=float)

    coverage_queue = informative_indices.copy()
    rng.shuffle(coverage_queue)

    for feat_idx in range(config.n_features):
        numerator_size, denominator_size = _choose_group_sizes(config.k, rng)
        chosen: List[int] = []

        def select_unique(target_list: List[int], required: int) -> List[int]:
            while len(target_list) < required:
                candidate = _pop_unique(coverage_queue, target_list + chosen)
                if candidate is None:
                    candidate = int(rng.choice(informative_indices))
                if candidate not in target_list and candidate not in chosen:
                    target_list.append(candidate)
            return target_list

        numerator_indices = select_unique([], numerator_size)
        chosen.extend(numerator_indices)
        denominator_indices = select_unique([], denominator_size)
        chosen.extend(denominator_indices)

        numerator_cols = [X.columns[idx] for idx in numerator_indices]
        denominator_cols = [X.columns[idx] for idx in denominator_indices]

        numerator_weights = _draw_weights(rng, config.numerator_weight_pool, size=len(numerator_indices))
        denominator_weights = _draw_weights(rng, config.denominator_weight_pool, size=len(denominator_indices))

        for col in numerator_cols:
            attribute_usage[col] = attribute_usage.get(col, 0) + 1
        for col in denominator_cols:
            attribute_usage[col] = attribute_usage.get(col, 0) + 1

        numerator_vals = (X.iloc[:, numerator_indices].to_numpy() * numerator_weights).sum(axis=1)
        denominator_inputs = np.abs(X.iloc[:, denominator_indices].to_numpy())
        denominator_vals = (denominator_inputs * denominator_weights).sum(axis=1) + config.epsilon

        raw_features[:, feat_idx] = numerator_vals / denominator_vals

        feature_defs.append(
            {
                "type": "ratioofsum",
                "numerator": {col: float(w) for col, w in zip(numerator_cols, numerator_weights)},
                "denominator": {col: float(w) for col, w in zip(denominator_cols, denominator_weights)},
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

    return Tier3Outputs(
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
    config: Tier3Config,
    rng: np.random.Generator,
) -> Tier3Outputs:
    if config.k < 3 or config.k > 4:
        raise ValueError(f"Tier3 sum-product expects k in [3,4], received {config.k}")
    if len(informative_indices) < config.k:
        raise ValueError("Number of informative attributes must be >= k for Tier3 sum-product")

    n_rows = X.shape[0]
    feature_defs: List[Dict[str, object]] = []
    attribute_usage: Dict[str, int] = {X.columns[idx]: 0 for idx in informative_indices}
    raw_features = np.zeros((n_rows, config.n_features), dtype=float)

    coverage_queue = informative_indices.copy()
    rng.shuffle(coverage_queue)

    for feat_idx in range(config.n_features):
        structure, sizes = _choose_sum_product_structure(config.k, rng)
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
        right_weights = None
        if structure == "sum_times_sum":
            right_weights = _draw_weights(rng, config.sum_weight_pool, size=len(right_indices))

        for col in left_cols:
            attribute_usage[col] = attribute_usage.get(col, 0) + 1
        for col in right_cols:
            attribute_usage[col] = attribute_usage.get(col, 0) + 1

        left_vals = (X.iloc[:, left_indices].to_numpy() * left_weights).sum(axis=1)
        if structure == "sum_times_single":
            single_col = right_cols[0]
            single_idx = right_indices[0]
            single_weight = _draw_weights(rng, config.sum_weight_pool, size=1)[0]
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
            assert right_weights is not None
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

    return Tier3Outputs(
        features=features_df,
        labels=labels,
        feature_defs=feature_defs,
        betas=betas.tolist(),
        intercept=float(intercept),
        attribute_usage=attribute_usage,
    )


def generate_tier3(
    X: pd.DataFrame,
    informative_indices: List[int],
    config: Tier3Config,
    rng: np.random.Generator,
) -> Tier3Outputs:
    spec = config.spec.lower()
    if spec == "ratioofsum":
        return _generate_ratio_of_sums(X, informative_indices, config, rng)
    if spec == "sumproduct":
        return _generate_sum_product(X, informative_indices, config, rng)
    raise ValueError(f"Unsupported Tier3 spec: {config.spec}")
