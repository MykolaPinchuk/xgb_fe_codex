from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .utils import sample_logistic_labels, standardize_columns


DEFAULT_WEIGHT_POOL = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])


@dataclass
class Tier1Config:
    positive_rate: float
    sigma_logit: float
    k: int
    n_features: int = 20
    weight_pool: Sequence[float] = tuple(DEFAULT_WEIGHT_POOL)


@dataclass
class Tier1Outputs:
    features: pd.DataFrame
    labels: pd.Series
    feature_defs: List[Dict[str, float]]
    betas: List[float]
    intercept: float
    attribute_usage: Dict[str, int]


def _draw_weight_vector(rng: np.random.Generator, pool: Sequence[float], size: int) -> np.ndarray:
    weights = rng.choice(pool, size=size, replace=True)
    return weights.astype(float)


def generate_tier1(
    X: pd.DataFrame,
    informative_indices: List[int],
    config: Tier1Config,
    rng: np.random.Generator,
) -> Tier1Outputs:
    if config.k < 2 or config.k > 6:
        raise ValueError(f"Tier1 expects k in [2,6], received {config.k}")
    if len(informative_indices) < config.k:
        raise ValueError("Number of informative attributes must be >= k")

    n_rows = X.shape[0]
    k = config.k

    coverage_queue = informative_indices.copy()
    rng.shuffle(coverage_queue)

    feature_defs: List[Dict[str, float]] = []
    attribute_usage: Dict[str, int] = {X.columns[idx]: 0 for idx in informative_indices}
    raw_features = np.zeros((n_rows, config.n_features), dtype=float)

    for feat_idx in range(config.n_features):
        chosen_set = []

        def pop_coverage() -> int | None:
            while coverage_queue:
                candidate = coverage_queue.pop()
                if candidate not in chosen_set:
                    return candidate
            return None

        anchor = pop_coverage()
        if anchor is None:
            anchor = int(rng.choice(informative_indices))
        chosen_set.append(anchor)

        while len(chosen_set) < k:
            candidate = pop_coverage()
            if candidate is not None:
                chosen_set.append(candidate)
                continue
            candidate = int(rng.choice(informative_indices))
            if candidate not in chosen_set:
                chosen_set.append(candidate)

        chosen_indices = np.array(sorted(chosen_set))

        chosen_cols = [X.columns[i] for i in chosen_indices]
        weights = _draw_weight_vector(rng, config.weight_pool, size=chosen_indices.size)

        for col in chosen_cols:
            attribute_usage[col] = attribute_usage.get(col, 0) + 1

        feature_vals = X.iloc[:, chosen_indices].to_numpy() @ weights
        raw_features[:, feat_idx] = feature_vals
        feature_defs.append({col: float(w) for col, w in zip(chosen_cols, weights)})

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

    return Tier1Outputs(
        features=features_df,
        labels=labels,
        feature_defs=feature_defs,
        betas=betas.tolist(),
        intercept=float(intercept),
        attribute_usage=attribute_usage,
    )
