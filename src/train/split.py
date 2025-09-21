from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class SplitConfig:
    val_frac: float
    seed: int


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    config: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.val_frac, random_state=config.seed)
    train_idx, val_idx = next(splitter.split(X, y))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    return X_train, X_val, y_train, y_val, train_idx, val_idx
