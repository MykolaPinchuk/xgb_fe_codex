from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from dgp.attributes import AttributeSpec, generate_attributes
from dgp.features_tier0 import Tier0Config, generate_tier0
from train.run_arm import ArmConfig, run_training_arm
from train.split import SplitConfig, stratified_split


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r") as fp:
        return yaml.safe_load(fp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XGB feature emulation experiment")
    parser.add_argument("--tier", type=str, default="tier0", choices=["tier0"], help="Tier to run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--config", type=Path, default=Path("config/defaults.yaml"), help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    seed = int(config.get("seed", 42))
    total_timeout = float(config.get("optuna", {}).get("timeout_seconds", 300))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("artifacts") / timestamp / args.tier
    run_dir.mkdir(parents=True, exist_ok=True)

    attr_spec = AttributeSpec(
        n_rows=int(config.get("n_rows", 10000)),
        n_attrs=int(config.get("n_attrs", 60)),
        n_informative=int(config.get("n_informative", 24)),
        seed=seed,
    )
    X, metadata, rng = generate_attributes(attr_spec)

    tier0_cfg = Tier0Config(
        positive_rate=float(config.get("positive_rate", 0.1)),
        sigma_logit=float(config.get("sigma_logit", 0.5)),
    )
    tier0_outputs = generate_tier0(X, metadata.informative_indices, tier0_cfg, rng)

    y = tier0_outputs.labels
    y.index = X.index

    split_cfg = SplitConfig(val_frac=float(config.get("val_frac", 0.2)), seed=seed)
    X_train_attr, X_val_attr, y_train, y_val, train_idx, val_idx = stratified_split(X, y, split_cfg)

    oracle_features = tier0_outputs.features
    oracle_train = oracle_features.iloc[train_idx]
    oracle_val = oracle_features.iloc[val_idx]

    xgb_cfg = config.get("xgb", {})
    optuna_cfg = config.get("optuna", {})

    summary = {
        "config": config,
        "selected_columns": tier0_outputs.selected_columns,
        "weights": tier0_outputs.weights,
    }
    (run_dir / "config_used.json").write_text(json.dumps(summary, indent=2))

    start_time = time.perf_counter()
    results: Dict[str, Dict[str, object]] = {}
    arms: List[Tuple[str, Tuple[pd.DataFrame, pd.DataFrame]]] = [
        ("ATTR", (X_train_attr, X_val_attr)),
        ("FEORACLE", (oracle_train, oracle_val)),
    ]

    for idx, (arm_name, (X_train, X_val)) in enumerate(arms):
        elapsed = time.perf_counter() - start_time
        remaining = max(total_timeout - elapsed, 10.0)
        arms_left = len(arms) - idx
        arm_timeout = max(remaining / arms_left, 10.0)
        arm_config = ArmConfig(
            xgb_params=xgb_cfg,
            n_trials=int(optuna_cfg.get("n_trials", 15)),
            early_stopping_rounds=int(xgb_cfg.get("early_stopping_rounds", 100)),
            timeout_seconds=arm_timeout,
            scale_pos_weight=float(xgb_cfg.get("scale_pos_weight", 2.0)),
        )
        arm_dir = run_dir / arm_name.lower()
        arm_result = run_training_arm(
            arm_name=arm_name,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            config=arm_config,
            seed=seed,
            artifacts_dir=arm_dir,
        )
        results[arm_name] = {
            "metrics": arm_result.metrics,
            "best_iteration": arm_result.best_iteration,
            "fit_seconds": arm_result.fit_seconds,
        }

    total_elapsed = time.perf_counter() - start_time
    (run_dir / "summary.json").write_text(json.dumps({"arms": results, "total_seconds": total_elapsed}, indent=2))

    print(json.dumps({"run_dir": str(run_dir), "total_seconds": total_elapsed, "arms": results}, indent=2))


if __name__ == "__main__":
    main()
