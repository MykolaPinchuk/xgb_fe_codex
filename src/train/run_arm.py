from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from .metrics import compute_metrics


@dataclass
class ArmConfig:
    xgb_params: Dict[str, Any]
    n_trials: int
    early_stopping_rounds: int
    timeout_seconds: float
    scale_pos_weight: float


@dataclass
class ArmResult:
    metrics: Dict[str, float]
    best_iteration: int
    best_params: Dict[str, Any]
    fit_seconds: float


def run_training_arm(
    arm_name: str,
    X_train,
    X_val,
    y_train,
    y_val,
    config: ArmConfig,
    seed: int,
    artifacts_dir: Path,
) -> ArmResult:
    imputer = SimpleImputer(strategy="median")
    X_train_np = imputer.fit_transform(X_train)
    X_val_np = imputer.transform(X_val)
    y_train_np = np.asarray(y_train)
    y_val_np = np.asarray(y_val)

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "tree_method": config.xgb_params.get("tree_method", "hist"),
            "max_bin": config.xgb_params.get("max_bin", 256),
            "max_depth": config.xgb_params.get("max_depth", 6),
            "learning_rate": trial.suggest_float("learning_rate", *config.xgb_params["learning_rate_range"]),
            "min_child_weight": trial.suggest_float("min_child_weight", *config.xgb_params["min_child_weight_range"]),
            "subsample": trial.suggest_float("subsample", *config.xgb_params["subsample_range"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *config.xgb_params["colsample_bytree_range"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *config.xgb_params["reg_alpha_range"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *config.xgb_params["reg_lambda_range"]),
        }

        model = XGBClassifier(
            n_estimators=1000,
            objective="binary:logistic",
            eval_metric="aucpr",
            early_stopping_rounds=config.early_stopping_rounds,
            n_jobs=-1,
            random_state=seed + trial.number,
            scale_pos_weight=config.scale_pos_weight,
            **params,
        )

        start = time.perf_counter()
        model.fit(
            X_train_np,
            y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            verbose=False,
        )
        fit_time = time.perf_counter() - start

        proba = model.predict_proba(X_val_np)[:, 1]
        metrics = compute_metrics(y_val_np, proba)
        best_iteration = getattr(model, "best_iteration", model.n_estimators)

        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("best_iteration", int(best_iteration))
        trial.set_user_attr("fit_seconds", fit_time)
        trial.set_user_attr("best_params", params)

        return metrics["pr_auc"]

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout_seconds,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    if study.best_trial is None:
        raise RuntimeError(f"Optuna failed to complete any trials for arm {arm_name}")

    best_trial = study.best_trial
    metrics = best_trial.user_attrs["metrics"]
    best_iteration = best_trial.user_attrs["best_iteration"]
    fit_seconds = best_trial.user_attrs["fit_seconds"]
    params = best_trial.user_attrs["best_params"]

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / f"metrics_{arm_name}.json").write_text(json.dumps(metrics, indent=2))
    (artifacts_dir / f"best_params_{arm_name}.json").write_text(json.dumps(params, indent=2))
    (artifacts_dir / f"timing_{arm_name}.json").write_text(json.dumps({"fit_seconds": fit_seconds}, indent=2))

    return ArmResult(metrics=metrics, best_iteration=best_iteration, best_params=params, fit_seconds=fit_seconds)
