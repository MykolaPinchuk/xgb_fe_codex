from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from dgp.attributes import AttributeMetadata, AttributeSpec, generate_attributes
from dgp.features_tier0 import Tier0Config, generate_tier0
from dgp.features_t1 import Tier1Config, generate_tier1
from dgp.features_t2 import Tier2Config, generate_tier2
from dgp.features_t3 import Tier3Config, generate_tier3
from dgp.features_t4 import Tier4Config, generate_tier4
from train.run_arm import ArmConfig, run_training_arm
from train.split import SplitConfig, stratified_split


def _attribute_diagnostics(X: pd.DataFrame, metadata: AttributeMetadata) -> Dict[str, Dict[str, float]]:
    variances = X.var(axis=0, ddof=0)
    diag: Dict[str, Dict[str, float]] = {
        "variance": {
            "min": float(variances.min()),
            "median": float(variances.median()),
            "max": float(variances.max()),
        }
    }

    informative_cols = [X.columns[i] for i in metadata.informative_indices]
    if len(informative_cols) > 1:
        corr = X[informative_cols].corr().to_numpy()
        mask = ~np.eye(len(informative_cols), dtype=bool)
        abs_corr = np.abs(corr[mask])
        diag["informative_abs_corr"] = {
            "mean": float(abs_corr.mean()),
            "max": float(abs_corr.max()),
        }

    if metadata.positive_only_indices:
        pos_cols = [X.columns[i] for i in metadata.positive_only_indices]
        pos_subset = X[pos_cols]
        pos_variances = pos_subset.var(axis=0, ddof=0)
        diag["positive_only"] = {
            "variance_min": float(pos_variances.min()),
            "variance_median": float(pos_variances.median()),
            "variance_max": float(pos_variances.max()),
            "mean": float(pos_subset.mean().mean()),
        }

    return diag


def _tier2_ratio_diagnostics(
    X: pd.DataFrame,
    feature_defs: List[Dict[str, object]],
    metadata: AttributeMetadata,
) -> Dict[str, object]:
    positive_set = {X.columns[i] for i in metadata.positive_only_indices}
    epsilon = 1e-3
    stats: List[Dict[str, float]] = []
    for feature_idx, definition in enumerate(feature_defs):
        if definition.get("type") != "ratio":
            continue
        denominator_col = definition.get("denominator")
        if denominator_col is None:
            continue
        denom_vals = X[denominator_col].to_numpy()
        if denominator_col not in positive_set:
            denom_vals = np.abs(denom_vals)
        denom_vals = denom_vals + epsilon
        stats.append(
            {
                "feature": f"z{feature_idx}",
                "min": float(denom_vals.min()),
                "median": float(np.median(denom_vals)),
                "std": float(denom_vals.std()),
                "p01": float(np.quantile(denom_vals, 0.01)),
                "p99": float(np.quantile(denom_vals, 0.99)),
            }
        )

    if not stats:
        return {}

    return {
        "ratio_denominators": stats,
    }


def _ratio_of_sums_diagnostics(
    X: pd.DataFrame,
    feature_defs: List[Dict[str, object]],
    epsilon: float,
) -> Dict[str, object]:
    stats: List[Dict[str, float]] = []
    for feature_idx, definition in enumerate(feature_defs):
        if definition.get("type") != "ratioofsum":
            continue
        denominator = definition.get("denominator", {})
        if not denominator:
            continue
        denom_vals = np.zeros(X.shape[0], dtype=float)
        for col, weight in denominator.items():
            denom_vals += np.abs(X[col].to_numpy()) * float(weight)
        denom_vals = denom_vals + float(epsilon)
        stats.append(
            {
                "feature": f"z{feature_idx}",
                "min": float(denom_vals.min()),
                "median": float(np.median(denom_vals)),
                "std": float(denom_vals.std()),
                "p01": float(np.quantile(denom_vals, 0.01)),
                "p99": float(np.quantile(denom_vals, 0.99)),
            }
        )

    if not stats:
        return {}

    return {
        "ratioofsum_denominators": stats,
    }


def collect_diagnostics(
    tier: str,
    tier_details: Dict[str, object],
    tier_outputs,
    X: pd.DataFrame,
    metadata: AttributeMetadata,
) -> Dict[str, object]:
    diagnostics: Dict[str, object] = {
        "attributes": _attribute_diagnostics(X, metadata)
    }

    feature_defs: List[Dict[str, object]] = tier_outputs.feature_defs if hasattr(tier_outputs, "feature_defs") else []

    if tier == "tier2" and tier_details.get("spec") == "ratio":
        diagnostics["tier_specific"] = _tier2_ratio_diagnostics(X, feature_defs, metadata)
    elif tier == "tier3" and tier_details.get("spec") == "ratioofsum":
        epsilon = tier_details.get("epsilon", 1e-3)
        diagnostics["tier_specific"] = _ratio_of_sums_diagnostics(X, feature_defs, epsilon)
    elif tier == "tier4" and tier_details.get("spec") == "ratioofsum":
        epsilon = tier_details.get("epsilon", 1e-3)
        diagnostics["tier_specific"] = _ratio_of_sums_diagnostics(X, feature_defs, epsilon)

    return diagnostics


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r") as fp:
        return yaml.safe_load(fp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XGB feature emulation experiment")
    parser.add_argument(
        "--tier",
        type=str,
        default="tier0",
        choices=["tier0", "tier1", "tier2", "tier3", "tier4"],
        help="Tier to run",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--config", type=Path, default=Path("config/defaults.yaml"), help="Config file path")
    parser.add_argument("--k", type=int, default=None, help="Feature arity for applicable tiers (e.g., tier1)")
    parser.add_argument("--spec", type=str, default=None, help="Tier-specific spec (e.g., product for tier2)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    seed = int(config.get("seed", 42))
    total_timeout = float(config.get("optuna", {}).get("timeout_seconds", 300))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    attribute_cfg = config.get("attribute", {}) or {}
    attr_spec_kwargs = {
        "n_rows": int(config.get("n_rows", 10000)),
        "n_attrs": int(config.get("n_attrs", 60)),
        "n_informative": int(config.get("n_informative", 24)),
        "seed": seed,
    }
    attr_spec_kwargs.update(attribute_cfg)
    attr_spec = AttributeSpec(**attr_spec_kwargs)
    X, metadata, rng = generate_attributes(attr_spec)

    positive_rate = float(config.get("positive_rate", 0.1))
    sigma_logit = float(config.get("sigma_logit", 0.5))

    if args.tier == "tier0":
        tier0_cfg = Tier0Config(
            positive_rate=positive_rate,
            sigma_logit=sigma_logit,
            n_label_attrs=int(config.get("n_label_attrs_tier0", 5)),
        )
        tier_outputs = generate_tier0(X, metadata.informative_indices, tier0_cfg, rng)
        oracle_features = tier_outputs.features
        tier_details = {
            "selected_columns": tier_outputs.selected_columns,
            "weights": tier_outputs.weights,
            "intercept": tier_outputs.intercept,
        }
        run_name = args.tier
    elif args.tier == "tier1":
        k_value = args.k if args.k is not None else 3
        n_features_cfg = config.get("n_true_features", {})
        n_features = int(n_features_cfg.get("tier1", 20))
        tier1_cfg = Tier1Config(
            positive_rate=positive_rate,
            sigma_logit=sigma_logit,
            k=int(k_value),
            n_features=n_features,
        )
        tier_outputs = generate_tier1(X, metadata.informative_indices, tier1_cfg, rng)
        oracle_features = tier_outputs.features
        tier_details = {
            "k": tier1_cfg.k,
            "n_features": tier1_cfg.n_features,
            "feature_defs": tier_outputs.feature_defs,
            "betas": tier_outputs.betas,
            "intercept": tier_outputs.intercept,
            "attribute_usage": {k: int(v) for k, v in tier_outputs.attribute_usage.items()},
        }
        run_name = f"{args.tier}_k{tier1_cfg.k}"
    elif args.tier == "tier2":
        spec_arg = (args.spec or "product").lower()
        supported_specs = {"product", "ratio", "absdiff", "minmax"}
        if spec_arg not in supported_specs:
            raise ValueError(f"Unsupported Tier2 spec '{spec_arg}'. Supported: {sorted(supported_specs)}")

        n_features_cfg = config.get("n_true_features", {})
        n_features = int(n_features_cfg.get("tier2", 20))
        tier2_cfg = Tier2Config(
            positive_rate=positive_rate,
            sigma_logit=sigma_logit,
            n_features=n_features,
            spec=spec_arg,
            positive_only_indices=metadata.positive_only_indices,
        )
        tier_outputs = generate_tier2(X, metadata.informative_indices, tier2_cfg, rng)
        oracle_features = tier_outputs.features
        tier_details = {
            "spec": spec_arg,
            "n_features": tier2_cfg.n_features,
            "feature_defs": tier_outputs.feature_defs,
            "betas": tier_outputs.betas,
            "intercept": tier_outputs.intercept,
            "attribute_usage": {k: int(v) for k, v in tier_outputs.attribute_usage.items()},
        }
        run_name = f"{args.tier}_{spec_arg}"
    elif args.tier == "tier3":
        spec_arg = (args.spec or "ratioofsum").lower()
        supported_specs = {"ratioofsum", "sumproduct"}
        if spec_arg not in supported_specs:
            raise ValueError(f"Unsupported Tier3 spec '{spec_arg}'. Supported: {sorted(supported_specs)}")

        k_value = args.k if args.k is not None else 3
        if k_value not in {3, 4}:
            raise ValueError("Tier3 ratio-of-sums expects k in {3, 4}")

        n_features_cfg = config.get("n_true_features", {})
        n_features = int(n_features_cfg.get("tier3", 20))
        tier3_cfg = Tier3Config(
            positive_rate=positive_rate,
            sigma_logit=sigma_logit,
            k=int(k_value),
            n_features=n_features,
            spec=spec_arg,
            positive_only_indices=metadata.positive_only_indices,
        )
        tier_outputs = generate_tier3(X, metadata.informative_indices, tier3_cfg, rng)
        oracle_features = tier_outputs.features
        tier_details = {
            "spec": spec_arg,
            "k": tier3_cfg.k,
            "n_features": tier3_cfg.n_features,
            "feature_defs": tier_outputs.feature_defs,
            "betas": tier_outputs.betas,
            "intercept": tier_outputs.intercept,
            "attribute_usage": {k: int(v) for k, v in tier_outputs.attribute_usage.items()},
            "epsilon": tier3_cfg.epsilon,
        }
        run_name = f"{args.tier}_{spec_arg}_k{tier3_cfg.k}"
    elif args.tier == "tier4":
        spec_arg = (args.spec or "ratioofsum").lower()
        supported_specs = {"ratioofsum", "sumproduct"}
        if spec_arg not in supported_specs:
            raise ValueError(f"Unsupported Tier4 spec '{spec_arg}'. Supported: {sorted(supported_specs)}")

        k_value = args.k if args.k is not None else 5
        if k_value not in {5, 6}:
            raise ValueError("Tier4 expects k in {5, 6}")

        n_features_cfg = config.get("n_true_features", {})
        n_features = int(n_features_cfg.get("tier4", 15))
        tier4_cfg = Tier4Config(
            positive_rate=positive_rate,
            sigma_logit=sigma_logit,
            k=int(k_value),
            n_features=n_features,
            spec=spec_arg,
            positive_only_indices=metadata.positive_only_indices,
        )
        tier_outputs = generate_tier4(X, metadata.informative_indices, tier4_cfg, rng)
        oracle_features = tier_outputs.features
        tier_details = {
            "spec": spec_arg,
            "k": tier4_cfg.k,
            "n_features": tier4_cfg.n_features,
            "feature_defs": tier_outputs.feature_defs,
            "betas": tier_outputs.betas,
            "intercept": tier_outputs.intercept,
            "attribute_usage": {k: int(v) for k, v in tier_outputs.attribute_usage.items()},
            "epsilon": tier4_cfg.epsilon,
        }
        run_name = f"{args.tier}_{spec_arg}_k{tier4_cfg.k}"
    else:
        raise ValueError(f"Unsupported tier: {args.tier}")

    run_dir = Path("artifacts") / timestamp / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = collect_diagnostics(args.tier, tier_details, tier_outputs, X, metadata)
    (run_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    y = tier_outputs.labels
    y.index = X.index

    split_cfg = SplitConfig(val_frac=float(config.get("val_frac", 0.2)), seed=seed)
    X_train_attr, X_val_attr, y_train, y_val, train_idx, val_idx = stratified_split(X, y, split_cfg)

    oracle_train = oracle_features.iloc[train_idx]
    oracle_val = oracle_features.iloc[val_idx]

    xgb_cfg = config.get("xgb", {})
    optuna_cfg = config.get("optuna", {})

    summary = {
        "config": config,
        "tier": args.tier,
        "run_name": run_name,
        "tier_details": tier_details,
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
