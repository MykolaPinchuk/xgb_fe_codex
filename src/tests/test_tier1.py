from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from dgp.attributes import AttributeSpec, generate_attributes
from dgp.features_t1 import Tier1Config, generate_tier1


def test_tier1_feature_properties():
    spec = AttributeSpec(n_rows=2000, n_attrs=30, n_informative=12, seed=99)
    X, metadata, rng = generate_attributes(spec)
    cfg = Tier1Config(positive_rate=0.1, sigma_logit=0.5, k=3, n_features=12)
    outputs = generate_tier1(X, metadata.informative_indices, cfg, rng)

    assert outputs.features.shape == (spec.n_rows, cfg.n_features)
    assert len(outputs.feature_defs) == cfg.n_features
    assert len(outputs.betas) == cfg.n_features

    means = outputs.features.mean().to_numpy()
    stds = outputs.features.std(ddof=0).to_numpy()

    assert np.allclose(means, 0.0, atol=1e-6)
    assert np.allclose(stds, 1.0, atol=1e-6)

    for feature_def in outputs.feature_defs:
        assert len(feature_def) == cfg.k

    used_attrs = set().union(*[feature_def.keys() for feature_def in outputs.feature_defs])
    expected_attrs = {f"x{i}" for i in metadata.informative_indices}
    assert expected_attrs.issubset(used_attrs)

    positive_rate = outputs.labels.mean()
    assert abs(positive_rate - cfg.positive_rate) < 0.02
