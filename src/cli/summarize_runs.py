from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class RunSummary:
    tier: str
    run_name: str
    path: Path
    extras: Dict[str, str]
    attr_metrics: Dict[str, float]
    oracle_metrics: Dict[str, float]
    attr_iteration: Optional[int]
    oracle_iteration: Optional[int]
    attr_fit_seconds: Optional[float]
    oracle_fit_seconds: Optional[float]

    @property
    def delta_pr_auc(self) -> Optional[float]:
        if "pr_auc" in self.attr_metrics and "pr_auc" in self.oracle_metrics:
            return self.oracle_metrics["pr_auc"] - self.attr_metrics["pr_auc"]
        return None


def _collect_run_dirs(root: Path) -> Iterable[Path]:
    for summary_path in root.rglob("summary.json"):
        yield summary_path.parent


def _load_json(path: Path) -> Dict:
    with path.open("r") as fp:
        return json.load(fp)


def _format_float(value: Optional[float], *, precision: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def _format_iter(value: Optional[int]) -> str:
    if value is None:
        return "-"
    return str(value)


def summarize_run(run_dir: Path) -> Optional[RunSummary]:
    summary_file = run_dir / "summary.json"
    config_file = run_dir / "config_used.json"
    if not summary_file.exists() or not config_file.exists():
        return None

    summary = _load_json(summary_file)
    config = _load_json(config_file)

    run_name = str(config.get("run_name", run_dir.name))
    tier = config.get("tier")
    if tier is None:
        tier_candidate = run_name.split("_")[0]
        tier = tier_candidate if tier_candidate.startswith("tier") else run_dir.parts[-1]
    tier = str(tier)

    arms = summary.get("arms", {})
    attr = arms.get("ATTR")
    oracle = arms.get("FEORACLE")
    if attr is None or oracle is None:
        return None

    extras: Dict[str, str] = {}
    tier_details = config.get("tier_details", {})
    for key in ("spec", "k", "n_features"):
        if key in tier_details:
            extras[key] = str(tier_details[key])

    return RunSummary(
        tier=tier,
        run_name=run_name,
        path=run_dir,
        extras=extras,
        attr_metrics=attr.get("metrics", {}),
        oracle_metrics=oracle.get("metrics", {}),
        attr_iteration=attr.get("best_iteration"),
        oracle_iteration=oracle.get("best_iteration"),
        attr_fit_seconds=attr.get("fit_seconds"),
        oracle_fit_seconds=oracle.get("fit_seconds"),
    )


def build_table(rows: List[RunSummary]) -> str:
    headers = [
        "Tier",
        "Run",
        "Spec",
        "ATTR_PR",
        "FE_PR",
        "ΔPR",
        "ATTR_ROC",
        "FE_ROC",
        "ATTR_Logloss",
        "FE_Logloss",
        "ATTR_iter",
        "FE_iter",
        "Path",
    ]

    lines = [" | ".join(headers)]
    lines.append(" | ".join("-" * len(h) for h in headers))

    for row in rows:
        spec = row.extras.get("spec") or row.extras.get("k") or "-"
        attr_pr = _format_float(row.attr_metrics.get("pr_auc"))
        oracle_pr = _format_float(row.oracle_metrics.get("pr_auc"))
        delta_pr = _format_float(row.delta_pr_auc)
        attr_roc = _format_float(row.attr_metrics.get("roc_auc"))
        oracle_roc = _format_float(row.oracle_metrics.get("roc_auc"))
        attr_ll = _format_float(row.attr_metrics.get("logloss"))
        oracle_ll = _format_float(row.oracle_metrics.get("logloss"))
        attr_iter = _format_iter(row.attr_iteration)
        oracle_iter = _format_iter(row.oracle_iteration)

        rel_path = row.path.as_posix()
        lines.append(
            " | ".join(
                [
                    row.tier,
                    row.run_name,
                    spec,
                    attr_pr,
                    oracle_pr,
                    delta_pr,
                    attr_roc,
                    oracle_roc,
                    attr_ll,
                    oracle_ll,
                    attr_iter,
                    oracle_iter,
                    rel_path,
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment runs")
    parser.add_argument("--root", type=Path, default=Path("artifacts"), help="Artifacts root directory")
    parser.add_argument("--tier", type=str, default=None, help="Filter by tier (e.g., tier2)")
    parser.add_argument("--latest-only", action="store_true", help="Keep only the latest run per (tier, run_name)")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"Artifacts root '{root}' does not exist")

    summaries: List[RunSummary] = []
    for run_dir in sorted(_collect_run_dirs(root)):
        summary = summarize_run(run_dir)
        if summary is None:
            continue
        if args.tier and summary.tier != args.tier:
            continue
        summaries.append(summary)

    if args.latest_only:
        latest: Dict[tuple[str, str], RunSummary] = {}
        for summary in summaries:
            key = (summary.tier, summary.run_name)
            existing = latest.get(key)
            if not existing or summary.path > existing.path:
                latest[key] = summary
        summaries = sorted(latest.values(), key=lambda s: (s.tier, s.run_name, s.path.as_posix()))
    else:
        summaries = sorted(summaries, key=lambda s: (s.tier, s.run_name, s.path.as_posix()))

    if not summaries:
        print("No runs found for the given criteria.")
        return

    print(build_table(summaries))


if __name__ == "__main__":
    main()
