from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import sample_logistic_labels


@dataclass
class Tier0Config:
    positive_rate: float
    sigma_logit: float
    n_label_attrs: int = 5


@dataclass
class Tier0Outputs:
    features: pd.DataFrame
    labels: pd.Series
    selected_columns: List[str]
    weights: List[float]
    intercept: float


def generate_tier0(X: pd.DataFrame, informative_indices: List[int], config: Tier0Config, rng: np.random.Generator) -> Tier0Outputs:
    n_label = min(config.n_label_attrs, len(informative_indices))
    chosen_idx = rng.choice(informative_indices, size=n_label, replace=False)
    chosen_cols = [X.columns[i] for i in chosen_idx]

    weights = rng.uniform(0.5, 1.5, size=n_label)
    signs = rng.choice([-1.0, 1.0], size=n_label)
    weights = weights * signs

    feature_matrix = X[chosen_cols].to_numpy()
    logits = feature_matrix @ weights
    labels, intercept, _ = sample_logistic_labels(
        logits=logits,
        positive_rate=config.positive_rate,
        sigma=config.sigma_logit,
        rng=rng,
    )

    features_df = X[chosen_cols].copy()
    return Tier0Outputs(
        features=features_df,
        labels=pd.Series(labels, name="y"),
        selected_columns=chosen_cols,
        weights=weights.tolist(),
        intercept=float(intercept),
    )
