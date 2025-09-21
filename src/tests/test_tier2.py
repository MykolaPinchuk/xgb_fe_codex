from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from dgp.attributes import AttributeSpec, generate_attributes
from dgp.features_t2 import Tier2Config, generate_tier2


def test_tier2_product_features():
    spec = AttributeSpec(n_rows=2500, n_attrs=40, n_informative=18, seed=77)
    X, metadata, rng = generate_attributes(spec)
    cfg = Tier2Config(positive_rate=0.1, sigma_logit=0.5, n_features=16, spec="product")

    outputs = generate_tier2(X, metadata.informative_indices, cfg, rng)

    assert outputs.features.shape == (spec.n_rows, cfg.n_features)
    assert len(outputs.feature_defs) == cfg.n_features
    assert len(outputs.betas) == cfg.n_features

    means = outputs.features.mean().to_numpy()
    stds = outputs.features.std(ddof=0).to_numpy()
    assert np.allclose(means, 0.0, atol=1e-6)
    assert np.allclose(stds, 1.0, atol=1e-6)

    for feature_def in outputs.feature_defs:
        assert feature_def.get("type") == "product"
        cols = feature_def.get("cols", [])
        assert len(cols) == 2

    attr_usage = outputs.attribute_usage
    assert all(count >= 1 for count in attr_usage.values())

    positive_rate = outputs.labels.mean()
    assert abs(positive_rate - cfg.positive_rate) < 0.02


def test_tier2_ratio_features():
    spec = AttributeSpec(n_rows=2500, n_attrs=60, n_informative=24, seed=33)
    X, metadata, rng = generate_attributes(spec)
    cfg = Tier2Config(positive_rate=0.1, sigma_logit=0.5, n_features=18, spec="ratio", positive_only_indices=metadata.positive_only_indices)

    outputs = generate_tier2(X, metadata.informative_indices, cfg, rng)

    assert outputs.features.shape == (spec.n_rows, cfg.n_features)
    assert len(outputs.feature_defs) == cfg.n_features

    means = outputs.features.mean().to_numpy()
    stds = outputs.features.std(ddof=0).to_numpy()
    assert np.allclose(means, 0.0, atol=1e-6)
    assert np.allclose(stds, 1.0, atol=1e-6)

    for feature_def in outputs.feature_defs:
        assert feature_def.get("type") == "ratio"
        assert "numerator" in feature_def and "denominator" in feature_def

    attr_usage = outputs.attribute_usage
    assert all(count >= 1 for count in attr_usage.values())

    positive_rate = outputs.labels.mean()
    assert abs(positive_rate - cfg.positive_rate) < 0.02
