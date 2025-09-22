from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class AttributeSpec:
    n_rows: int
    n_attrs: int
    n_informative: int
    seed: int
    block_size: int = 6
    rho: float = 0.4
    positive_fraction: float = 0.5
    scale_low: float = 0.6
    scale_high: float = 2.5
    lognormal_scale_low: float = 0.7
    lognormal_scale_high: float = 1.3
    lognormal_shift_mean: float = 0.25
    lognormal_shift_std: float = 0.15
    distractor_rho: float = 0.2
    distractor_scale_low: float = 0.5
    distractor_scale_high: float = 3.0
    heavy_tail_fraction: float = 0.35
    heavy_tail_df: float = 3.0


@dataclass
class AttributeMetadata:
    informative_indices: List[int]
    positive_only_indices: List[int]


def _make_block_covariance(size: int, rho: float) -> np.ndarray:
    clipped_rho = float(np.clip(rho, -0.95, 0.95))
    cov = np.full((size, size), clipped_rho)
    np.fill_diagonal(cov, 1.0)
    return cov


def _generate_informative_block(
    rng: np.random.Generator,
    n_rows: int,
    size: int,
    rho: float,
    scale_low: float,
    scale_high: float,
) -> np.ndarray:
    cov = _make_block_covariance(size, rho)
    raw = rng.multivariate_normal(mean=np.zeros(size), cov=cov, size=n_rows)
    scales = rng.uniform(scale_low, scale_high, size=size)
    return raw * scales


def _apply_positive_transform(
    rng: np.random.Generator,
    values: np.ndarray,
    scale_low: float,
    scale_high: float,
    shift_mean: float,
    shift_std: float,
) -> np.ndarray:
    if values.shape[1] == 0:
        return values
    # Standardize per column to keep variation controllable before exponentiating.
    col_means = values.mean(axis=0, keepdims=True)
    col_stds = values.std(axis=0, ddof=0, keepdims=True)
    col_stds[col_stds == 0] = 1.0
    standardized = (values - col_means) / col_stds
    scales = rng.uniform(scale_low, scale_high, size=values.shape[1])
    shifts = rng.normal(shift_mean, shift_std, size=values.shape[1])
    transformed = np.exp(standardized * scales + shifts)
    return transformed


def _generate_distractor_block(
    rng: np.random.Generator,
    n_rows: int,
    size: int,
    rho: float,
    scale_low: float,
    scale_high: float,
    heavy_tail_fraction: float,
    heavy_tail_df: float,
) -> np.ndarray:
    cov = _make_block_covariance(size, rho)
    base = rng.multivariate_normal(mean=np.zeros(size), cov=cov, size=n_rows)
    scales = rng.uniform(scale_low, scale_high, size=size)
    base *= scales

    if heavy_tail_fraction <= 0:
        return base

    heavy_mask = rng.uniform(size=size) < heavy_tail_fraction
    if not np.any(heavy_mask):
        return base

    heavy_count = int(np.sum(heavy_mask))
    heavy_samples = rng.standard_t(heavy_tail_df, size=(n_rows, heavy_count))
    heavy_samples *= scales[heavy_mask]
    base[:, heavy_mask] = heavy_samples
    return base


def generate_attributes(spec: AttributeSpec) -> tuple[pd.DataFrame, AttributeMetadata, np.random.Generator]:
    rng = np.random.default_rng(spec.seed)

    n_rows = spec.n_rows
    n_attrs = spec.n_attrs
    n_informative = spec.n_informative

    if n_informative > n_attrs:
        raise ValueError("n_informative cannot exceed n_attrs")

    X = np.zeros((n_rows, n_attrs), dtype=float)

    informative_indices = list(range(n_informative))
    positive_target = int(round(n_informative * spec.positive_fraction))
    if n_informative > 0:
        positive_target = min(max(0, positive_target), n_informative - 1 if n_informative > 1 else 1)
    positive_only = informative_indices[:positive_target]

    block_size = max(1, int(spec.block_size))

    # Build correlated informative blocks with heterogeneous scales.
    start = 0
    while start < n_informative:
        current_size = min(block_size, n_informative - start)
        block = _generate_informative_block(
            rng,
            n_rows,
            current_size,
            spec.rho,
            spec.scale_low,
            spec.scale_high,
        )
        X[:, start : start + current_size] = block
        start += current_size

    if positive_only:
        positive_cols = np.array(positive_only)
        transformed = _apply_positive_transform(
            rng,
            X[:, positive_cols],
            spec.lognormal_scale_low,
            spec.lognormal_scale_high,
            spec.lognormal_shift_mean,
            spec.lognormal_shift_std,
        )
        X[:, positive_cols] = transformed

    # Fill remaining distractor attributes with a mix of correlated and heavy-tailed signals.
    remaining = n_attrs - n_informative
    start_idx = n_informative
    offset = 0
    while offset < remaining:
        current_size = min(block_size, remaining - offset)
        block = _generate_distractor_block(
            rng,
            n_rows,
            current_size,
            spec.distractor_rho,
            spec.distractor_scale_low,
            spec.distractor_scale_high,
            spec.heavy_tail_fraction,
            spec.heavy_tail_df,
        )
        X[:, start_idx + offset : start_idx + offset + current_size] = block
        offset += current_size

    columns = [f"x{i}" for i in range(n_attrs)]
    df = pd.DataFrame(X, columns=columns)

    metadata = AttributeMetadata(
        informative_indices=informative_indices,
        positive_only_indices=list(positive_only),
    )
    return df, metadata, rng
