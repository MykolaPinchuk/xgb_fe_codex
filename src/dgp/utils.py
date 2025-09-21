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


def generate_noisy_logits(logits: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return logits
    noise = rng.normal(0.0, sigma, size=logits.shape[0])
    return logits + noise


def sample_logistic_labels(
    logits: np.ndarray,
    positive_rate: float,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray]:
    noisy_logits = generate_noisy_logits(logits, sigma, rng)
    intercept = find_intercept_for_target_rate(noisy_logits, positive_rate)
    probabilities = sigmoid(noisy_logits + intercept)
    draws = rng.uniform(0.0, 1.0, size=probabilities.shape[0])
    labels = (probabilities > draws).astype(int)
    return labels, intercept, probabilities
