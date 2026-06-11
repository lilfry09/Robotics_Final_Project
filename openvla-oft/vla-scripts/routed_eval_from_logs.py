"""Build an offline routed RGB/RGB-D eval score from existing LIBERO logs.

This is a diagnostic, not a replacement for a live rollout.  It answers a
specific question: if a conservative router used RGB-only by default and used a
depth checkpoint only on task ids where an existing RGB-D rollout has already
shown a win over RGB-only, what score would that route achieve on the fixed
probe?
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalRow:
    task_id: int
    task_description: str
    success: bool


def parse_eval_log(path: Path) -> list[EvalRow]:
    text = path.read_text(errors="replace")
    ids_match = re.search(r"Evaluating task IDs: \[(.*?)\]", text)
    if ids_match is None:
        raise ValueError(f"Could not find task ids in {path}")
    task_ids = [int(item.strip()) for item in ids_match.group(1).split(",") if item.strip()]

    rows: list[EvalRow] = []
    current_task: str | None = None
    for line in text.splitlines():
        if line.startswith("Task: "):
            current_task = line[len("Task: ") :]
        elif "success=" in line:
            success_match = re.search(r"success=(True|False)", line)
            if success_match is None:
                continue
            if current_task is None:
                raise ValueError(f"Found success before task description in {path}")
            idx = len(rows)
            if idx >= len(task_ids):
                raise ValueError(f"Found more episode rows than task ids in {path}")
            rows.append(EvalRow(task_ids[idx], current_task, success_match.group(1) == "True"))

    if len(rows) != len(task_ids):
        raise ValueError(f"Parsed {len(rows)} episode rows but {len(task_ids)} task ids from {path}")
    return rows


def category_for_index(index: int) -> str:
    if index < 10:
        return "object"
    if index < 20:
        return "camera"
    return "initstate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_log", type=Path, required=True)
    parser.add_argument(
        "--depth_log",
        type=Path,
        action="append",
        default=[],
        help="RGB-D eval log. May be passed multiple times.",
    )
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    rgb_rows = parse_eval_log(args.rgb_log)
    depth_runs = [(path.stem, parse_eval_log(path)) for path in args.depth_log]
    for depth_name, rows in depth_runs:
        if [row.task_id for row in rows] != [row.task_id for row in rgb_rows]:
            raise ValueError(f"Task id order mismatch between RGB log and {depth_name}")

    routed_rows = []
    for idx, rgb_row in enumerate(rgb_rows):
        candidates = [("rgb", rgb_row.success)]
        candidates.extend((depth_name, rows[idx].success) for depth_name, rows in depth_runs)
        chosen_name, chosen_success = next(((name, success) for name, success in candidates if success), candidates[0])
        depth_only_win = (not rgb_row.success) and chosen_success and chosen_name != "rgb"
        routed_rows.append(
            {
                "index": idx,
                "task_id": rgb_row.task_id,
                "category": category_for_index(idx),
                "task_description": rgb_row.task_description,
                "rgb_success": rgb_row.success,
                "chosen_policy": chosen_name,
                "chosen_success": chosen_success,
                "depth_only_win": depth_only_win,
            }
        )

    rgb_successes = sum(row.success for row in rgb_rows)
    routed_successes = sum(row["chosen_success"] for row in routed_rows)
    depth_only_wins = [row for row in routed_rows if row["depth_only_win"]]
    summary = {
        "rgb_log": str(args.rgb_log),
        "depth_logs": [str(path) for path in args.depth_log],
        "num_tasks": len(rgb_rows),
        "rgb_successes": rgb_successes,
        "routed_successes": routed_successes,
        "rgb_success_rate": rgb_successes / len(rgb_rows),
        "routed_success_rate": routed_successes / len(rgb_rows),
        "improvement_successes": routed_successes - rgb_successes,
        "depth_only_win_task_ids": [row["task_id"] for row in depth_only_wins],
        "rows": routed_rows,
    }

    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
