from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class AttributeSpec:
    n_rows: int
    n_attrs: int
    n_informative: int
    seed: int


@dataclass
class AttributeMetadata:
    informative_indices: List[int]
    positive_only_indices: List[int]


def generate_attributes(spec: AttributeSpec) -> tuple[pd.DataFrame, AttributeMetadata, np.random.Generator]:
    rng = np.random.default_rng(spec.seed)
    X = rng.normal(0.0, 1.0, size=(spec.n_rows, spec.n_attrs))

    informative = list(range(spec.n_informative))
    positive_only = informative[: spec.n_informative // 2]
    if positive_only:
        X[:, positive_only] = np.exp(0.5 * rng.normal(size=(spec.n_rows, len(positive_only))))

    columns = [f"x{i}" for i in range(spec.n_attrs)]
    df = pd.DataFrame(X, columns=columns)

    metadata = AttributeMetadata(informative_indices=informative, positive_only_indices=list(positive_only))
    return df, metadata, rng
