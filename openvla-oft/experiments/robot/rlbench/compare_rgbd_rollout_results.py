"""Compare matched RLBench RGB-only and RGB-D rollout results.

This script is the final causal gate for the new RGB-D benchmark direction:

1. RGB-D with normal depth must beat the matched RGB-only baseline.
2. RGB-D with normal depth must also beat null and shuffled depth ablations.

Inputs can be either:

- BridgeVLA/RVT-style ``eval_results.csv`` with columns such as
  ``task`` and ``success rate``.
- JSON files with fields such as ``success_rate``, ``mean_success``,
  ``scores``, ``tasks`` or ``task_results``.

Exit code is 0 when the gate passes and 2 when it fails.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ResultSet:
    path: Path
    overall: float
    per_task: dict[str, float]


def _as_rate(value: Any) -> float:
    if isinstance(value, str):
        value = value.strip().replace("%", "")
    rate = float(value)
    if rate > 1.0:
        rate /= 100.0
    return rate


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list of values")
    return float(sum(values) / len(values))


def _load_csv(path: Path) -> ResultSet:
    per_task: dict[str, float] = {}
    overall_values: list[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        rate_key = None
        for candidate in ("success rate", "success_rate", "mean_success", "score"):
            if candidate in reader.fieldnames:
                rate_key = candidate
                break
        if rate_key is None:
            raise ValueError(f"CSV file is missing a success-rate column: {path}")
        task_key = "task" if "task" in reader.fieldnames else None

        for row in reader:
            rate = _as_rate(row[rate_key])
            overall_values.append(rate)
            if task_key is not None:
                per_task[str(row[task_key])] = rate

    return ResultSet(path=path, overall=_mean(overall_values), per_task=per_task)


def _extract_task_mapping(obj: Any) -> dict[str, float]:
    if not isinstance(obj, dict):
        return {}

    for key in ("task_results", "per_task", "tasks"):
        value = obj.get(key)
        if isinstance(value, dict):
            mapping = {}
            for task, raw in value.items():
                if isinstance(raw, dict):
                    for rate_key in ("success_rate", "success rate", "mean_success", "score"):
                        if rate_key in raw:
                            mapping[str(task)] = _as_rate(raw[rate_key])
                            break
                else:
                    mapping[str(task)] = _as_rate(raw)
            if mapping:
                return mapping

    return {}


def _load_json(path: Path) -> ResultSet:
    with path.open("r") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        values = [_as_rate(item) for item in obj]
        return ResultSet(path=path, overall=_mean(values), per_task={})

    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported JSON root type in {path}: {type(obj).__name__}")

    for key in ("success_rate", "success rate", "mean_success", "score", "overall"):
        if key in obj:
            overall = _as_rate(obj[key])
            return ResultSet(path=path, overall=overall, per_task=_extract_task_mapping(obj))

    per_task = _extract_task_mapping(obj)
    if per_task:
        return ResultSet(path=path, overall=_mean(list(per_task.values())), per_task=per_task)

    for key in ("scores", "successes", "returns"):
        value = obj.get(key)
        if isinstance(value, list):
            return ResultSet(path=path, overall=_mean([_as_rate(v) for v in value]), per_task={})

    raise ValueError(f"Could not find success metrics in JSON file: {path}")


def load_result(path: str | Path) -> ResultSet:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    if path.suffix.lower() == ".json":
        return _load_json(path)
    raise ValueError(f"Unsupported result file extension: {path.suffix}; expected .csv or .json")


def common_task_advantage(a: ResultSet, b: ResultSet) -> dict[str, float]:
    tasks = sorted(set(a.per_task) & set(b.per_task))
    return {task: a.per_task[task] - b.per_task[task] for task in tasks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_only", required=True, help="Matched RGB-only rollout result file.")
    parser.add_argument("--rgbd_normal", required=True, help="RGB-D rollout result file using normal depth.")
    parser.add_argument("--rgbd_null", required=True, help="RGB-D rollout result file using null depth.")
    parser.add_argument("--rgbd_shuffle", required=True, help="RGB-D rollout result file using shuffled/corrupt depth.")
    parser.add_argument("--min_rgb_gain", type=float, default=0.05, help="Required absolute success-rate gain over RGB-only.")
    parser.add_argument(
        "--min_ablation_gain",
        type=float,
        default=0.05,
        help="Required absolute success-rate gain over both null and shuffle depth.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path.")
    args = parser.parse_args()

    rgb = load_result(args.rgb_only)
    normal = load_result(args.rgbd_normal)
    null = load_result(args.rgbd_null)
    shuffle = load_result(args.rgbd_shuffle)

    gain_rgb = normal.overall - rgb.overall
    gain_null = normal.overall - null.overall
    gain_shuffle = normal.overall - shuffle.overall
    passed = (
        gain_rgb >= args.min_rgb_gain
        and gain_null >= args.min_ablation_gain
        and gain_shuffle >= args.min_ablation_gain
    )

    summary: dict[str, Any] = {
        "passed": passed,
        "thresholds": {
            "min_rgb_gain": args.min_rgb_gain,
            "min_ablation_gain": args.min_ablation_gain,
        },
        "overall": {
            "rgb_only": rgb.overall,
            "rgbd_normal": normal.overall,
            "rgbd_null": null.overall,
            "rgbd_shuffle": shuffle.overall,
        },
        "gains": {
            "normal_minus_rgb_only": gain_rgb,
            "normal_minus_null": gain_null,
            "normal_minus_shuffle": gain_shuffle,
        },
        "per_task_gains": {
            "normal_minus_rgb_only": common_task_advantage(normal, rgb),
            "normal_minus_null": common_task_advantage(normal, null),
            "normal_minus_shuffle": common_task_advantage(normal, shuffle),
        },
        "inputs": {
            "rgb_only": str(rgb.path),
            "rgbd_normal": str(normal.path),
            "rgbd_null": str(null.path),
            "rgbd_shuffle": str(shuffle.path),
        },
    }

    print("RLBench RGB-D causal rollout gate")
    print(f"  RGB-only     : {rgb.overall:.3f}")
    print(f"  RGB-D normal : {normal.overall:.3f}")
    print(f"  RGB-D null   : {null.overall:.3f}")
    print(f"  RGB-D shuffle: {shuffle.overall:.3f}")
    print(f"  normal - RGB-only: {gain_rgb:+.3f}  required >= {args.min_rgb_gain:.3f}")
    print(f"  normal - null    : {gain_null:+.3f}  required >= {args.min_ablation_gain:.3f}")
    print(f"  normal - shuffle : {gain_shuffle:+.3f}  required >= {args.min_ablation_gain:.3f}")
    print("  result:", "GO" if passed else "NO-GO")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote: {args.output}")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
