from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def find_intercept_for_target_rate(logits: np.ndarray, target_rate: float, *, tol: float = 1e-6, max_iter: int = 100) -> float:
    lower, upper = -20.0, 20.0
    for _ in range(max_iter):
        mid = (lower + upper) / 2.0
        rate = sigmoid(logits + mid).mean()
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            lower = mid
        else:
            upper = mid
    return (lower + upper) / 2.0


def standardize_columns(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    return (matrix - means) / stds


def ensure_unique_selection(pool: Sequence[int], rng: np.random.Generator, k: int) -> np.ndarray:
    return rng.choice(pool, size=k, replace=False)
