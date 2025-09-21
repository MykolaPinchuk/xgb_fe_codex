from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from dgp.attributes import AttributeSpec, generate_attributes
from dgp.features_tier0 import Tier0Config, generate_tier0


def test_tier0_positive_rate_close():
    spec = AttributeSpec(n_rows=2000, n_attrs=20, n_informative=10, seed=123)
    X, metadata, rng = generate_attributes(spec)
    cfg = Tier0Config(positive_rate=0.1, sigma_logit=0.4)
    outputs = generate_tier0(X, metadata.informative_indices, cfg, rng)

    positive_rate = outputs.labels.mean()
    assert abs(positive_rate - 0.1) < 0.02


def test_tier0_selected_columns_subset():
    spec = AttributeSpec(n_rows=100, n_attrs=12, n_informative=6, seed=7)
    X, metadata, rng = generate_attributes(spec)
    cfg = Tier0Config(positive_rate=0.1, sigma_logit=0.5)
    outputs = generate_tier0(X, metadata.informative_indices, cfg, rng)

    for col in outputs.selected_columns:
        assert col in X.columns
        assert np.allclose(outputs.features[col], X[col])
