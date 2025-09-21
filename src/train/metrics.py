from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    ll = log_loss(y_true, np.clip(y_proba, 1e-8, 1 - 1e-8))
    return {"pr_auc": float(pr_auc), "roc_auc": float(roc_auc), "logloss": float(ll)}
