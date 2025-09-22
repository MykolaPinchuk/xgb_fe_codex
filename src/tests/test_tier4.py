from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from dgp.attributes import AttributeSpec, generate_attributes
from dgp.features_t4 import Tier4Config, generate_tier4


def _assert_standardized(features_df):
    means = features_df.mean().to_numpy()
    stds = features_df.std(ddof=0).to_numpy()
    assert np.allclose(means, 0.0, atol=1e-6)
    assert np.allclose(stds, 1.0, atol=1e-6)


def test_tier4_ratioofsum_k6():
    spec = AttributeSpec(n_rows=3200, n_attrs=60, n_informative=24, seed=515)
    X, metadata, rng = generate_attributes(spec)
    cfg = Tier4Config(positive_rate=0.1, sigma_logit=0.45, k=6, n_features=15, spec="ratioofsum")

    outputs = generate_tier4(X, metadata.informative_indices, cfg, rng)

    assert outputs.features.shape == (spec.n_rows, cfg.n_features)
    assert len(outputs.feature_defs) == cfg.n_features

    _assert_standardized(outputs.features)

    for feature_def in outputs.feature_defs:
        assert feature_def.get("type") == "ratioofsum"
        numerator = feature_def.get("numerator", {})
        denominator = feature_def.get("denominator", {})
        assert 2 <= len(numerator) <= 3
        assert len(numerator) + len(denominator) == cfg.k
        assert all(weight != 0 for weight in numerator.values())
        assert all(weight > 0 for weight in denominator.values())
        cv = feature_def.get("denominator_cv")
        assert cv is not None
        assert cv > 0.1

    assert all(count >= 1 for count in outputs.attribute_usage.values())
    assert abs(outputs.labels.mean() - cfg.positive_rate) < 0.02


def test_tier4_sumproduct_k6():
    spec = AttributeSpec(n_rows=3400, n_attrs=60, n_informative=24, seed=626)
    X, metadata, rng = generate_attributes(spec)
    cfg = Tier4Config(positive_rate=0.1, sigma_logit=0.45, k=6, n_features=15, spec="sumproduct")

    outputs = generate_tier4(X, metadata.informative_indices, cfg, rng)

    assert outputs.features.shape == (spec.n_rows, cfg.n_features)
    _assert_standardized(outputs.features)

    for feature_def in outputs.feature_defs:
        assert feature_def.get("type") == "sumproduct"
        structure = feature_def.get("structure")
        left = feature_def.get("left", {})
        right = feature_def.get("right", {})
        if structure == "sum_times_single":
            assert len(left) == 5
            assert len(right) == 1
        else:
            assert structure == "sum_times_sum"
            assert len(left) in {3, 4}
            assert len(right) == cfg.k - len(left)
        assert len(left) + len(right) == cfg.k

    assert all(count >= 1 for count in outputs.attribute_usage.values())
    assert abs(outputs.labels.mean() - cfg.positive_rate) < 0.02
