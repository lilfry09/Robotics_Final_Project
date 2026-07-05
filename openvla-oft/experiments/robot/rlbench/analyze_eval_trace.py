#!/usr/bin/env python3
"""Analyze RLBench DepthVLA per-step eval traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No trace rows found in {path}")
    return rows


def arr(row: dict[str, Any], key: str, dims: int | None = None) -> np.ndarray | None:
    value = row.get(key)
    if value is None:
        return None
    out = np.asarray(value, dtype=np.float32)
    if dims is not None and out.shape[-1] < dims:
        return None
    return out


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))


def safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def safe_min(values: list[float]) -> float | None:
    return float(np.min(values)) if values else None


def safe_max(values: list[float]) -> float | None:
    return float(np.max(values)) if values else None


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return None
    return float(np.dot(a, b) / denom)


def trace_metrics(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    depth_distances: list[float] = []
    target_depth_distances: list[float] = []
    delta_depth_cosines: list[float] = []
    waypoint_norms: list[float] = []
    waypoint_saturation = 0
    waypoint_components = 0
    grippers: list[float] = []
    delta_norms: list[float] = []
    target_distances: list[float] = []
    post_close_pull_steps = 0
    latched_depth_point_action_steps = 0
    latched_depth_point_distances: list[float] = []

    for row in rows:
        ee = arr(row, "ee_xyz_before", 3)
        target = arr(row, "target_xyz", 3)
        depth_point = arr(row, "depth_point_xyz", 3)
        delta = arr(row, "delta_action")
        waypoint = arr(row, "depth_waypoint_chunk_xyz_action", 3)

        if ee is not None and depth_point is not None:
            vector_to_depth = depth_point[:3] - ee[:3]
            depth_distances.append(float(np.linalg.norm(vector_to_depth)))
            if delta is not None:
                cos = cosine(delta[:3], vector_to_depth)
                if cos is not None:
                    delta_depth_cosines.append(cos)
        if target is not None and depth_point is not None:
            target_depth_distances.append(l2(target[:3], depth_point[:3]))
        if waypoint is not None:
            waypoint_norms.append(float(np.linalg.norm(waypoint[:3])))
            waypoint_components += int(np.prod(waypoint[:3].shape))
            waypoint_saturation += int(np.sum(np.abs(waypoint[:3]) >= 0.399))
        if row.get("gripper_command") is not None:
            grippers.append(float(row["gripper_command"]))
        if row.get("delta_xyz_norm") is not None:
            delta_norms.append(float(row["delta_xyz_norm"]))
        if row.get("target_distance") is not None:
            target_distances.append(float(row["target_distance"]))
        if row.get("post_close_pull_active"):
            post_close_pull_steps += 1
        if row.get("latched_depth_point_action_active"):
            latched_depth_point_action_steps += 1
        if row.get("latched_depth_point_distance") is not None:
            latched_depth_point_distances.append(float(row["latched_depth_point_distance"]))

    first = rows[0]
    last = rows[-1]
    first_ee = arr(first, "ee_xyz_before", 3)
    last_ee = arr(last, "ee_xyz_before", 3)
    ee_displacement = l2(first_ee[:3], last_ee[:3]) if first_ee is not None and last_ee is not None else None
    first_depth = arr(first, "depth_point_xyz", 3)
    last_depth = arr(last, "depth_point_xyz", 3)
    selected_point_drift = l2(first_depth[:3], last_depth[:3]) if first_depth is not None and last_depth is not None else None

    close_steps = [int(row["step"]) for row in rows if float(row.get("gripper_command", 1.0)) < 0.5]
    return {
        "name": name,
        "rows": len(rows),
        "depth_mode": first.get("depth_mode"),
        "task": first.get("task"),
        "episode": first.get("episode"),
        "step_first": first.get("step"),
        "step_last": last.get("step"),
        "action_chunk_exec_horizon": first.get("action_chunk_exec_horizon"),
        "new_prediction_steps": [int(row["step"]) for row in rows if row.get("new_prediction")],
        "chunk_indices_first16": [int(row["chunk_index"]) for row in rows[:16]],
        "depth_point_rows": sum(row.get("depth_point_xyz") is not None for row in rows),
        "waypoint_rows": sum(row.get("depth_waypoint_chunk_xyz_action") is not None for row in rows),
        "ee_displacement": ee_displacement,
        "selected_point_drift": selected_point_drift,
        "ee_to_depth_point_first": depth_distances[0] if depth_distances else None,
        "ee_to_depth_point_last": depth_distances[-1] if depth_distances else None,
        "ee_to_depth_point_min": safe_min(depth_distances),
        "ee_to_depth_point_mean": safe_mean(depth_distances),
        "target_to_depth_point_mean": safe_mean(target_depth_distances),
        "delta_toward_depth_cosine_mean": safe_mean(delta_depth_cosines),
        "delta_toward_depth_cosine_first": delta_depth_cosines[0] if delta_depth_cosines else None,
        "delta_xyz_norm_mean": safe_mean(delta_norms),
        "delta_xyz_norm_max": safe_max(delta_norms),
        "target_distance_mean": safe_mean(target_distances),
        "waypoint_norm_mean": safe_mean(waypoint_norms),
        "waypoint_saturation_fraction": (
            float(waypoint_saturation / waypoint_components) if waypoint_components else None
        ),
        "gripper_mean": safe_mean(grippers),
        "gripper_min": safe_min(grippers),
        "first_close_step": close_steps[0] if close_steps else None,
        "post_close_pull_steps": post_close_pull_steps,
        "depth_point_latch_mode": first.get("depth_point_latch_mode"),
        "latched_depth_point_action_steps": latched_depth_point_action_steps,
        "latched_depth_point_distance_first": (
            latched_depth_point_distances[0] if latched_depth_point_distances else None
        ),
        "latched_depth_point_distance_last": (
            latched_depth_point_distances[-1] if latched_depth_point_distances else None
        ),
        "latched_depth_point_distance_min": safe_min(latched_depth_point_distances),
    }


def pair_metrics(name_a: str, rows_a: list[dict[str, Any]], name_b: str, rows_b: list[dict[str, Any]]) -> dict[str, Any]:
    first_a, first_b = rows_a[0], rows_b[0]
    metrics: dict[str, Any] = {"pair": f"{name_a}_vs_{name_b}"}
    for key in ("depth_point_xyz", "depth_waypoint_chunk_xyz_action"):
        a = arr(first_a, key, 3)
        b = arr(first_b, key, 3)
        if a is not None and b is not None:
            metrics[f"first_{key}_l2"] = l2(a[:3], b[:3])
    delta_a = arr(first_a, "delta_action")
    delta_b = arr(first_b, "delta_action")
    if delta_a is not None and delta_b is not None:
        metrics["first_delta_xyz_l2"] = l2(delta_a[:3], delta_b[:3])

    common = min(len(rows_a), len(rows_b))
    step_delta_l2: list[float] = []
    step_point_l2: list[float] = []
    for idx in range(common):
        da = arr(rows_a[idx], "delta_action")
        db = arr(rows_b[idx], "delta_action")
        pa = arr(rows_a[idx], "depth_point_xyz", 3)
        pb = arr(rows_b[idx], "depth_point_xyz", 3)
        if da is not None and db is not None:
            step_delta_l2.append(l2(da[:3], db[:3]))
        if pa is not None and pb is not None:
            step_point_l2.append(l2(pa[:3], pb[:3]))
    metrics["common_steps"] = common
    metrics["mean_delta_xyz_l2_common_steps"] = safe_mean(step_delta_l2)
    metrics["mean_depth_point_l2_common_steps"] = safe_mean(step_point_l2)
    return metrics


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def write_markdown(output: Path, metrics: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# RLBench Eval Trace Analysis", ""]
    for item in metrics:
        lines.extend(
            [
                f"## {item['name']}",
                "",
                "| metric | value |",
                "|---|---:|",
            ]
        )
        for key in (
            "rows",
            "depth_mode",
            "step_last",
            "action_chunk_exec_horizon",
            "new_prediction_steps",
            "ee_displacement",
            "selected_point_drift",
            "ee_to_depth_point_first",
            "ee_to_depth_point_last",
            "ee_to_depth_point_min",
            "delta_toward_depth_cosine_mean",
            "delta_xyz_norm_mean",
            "waypoint_saturation_fraction",
            "gripper_mean",
            "gripper_min",
            "first_close_step",
            "post_close_pull_steps",
            "depth_point_latch_mode",
            "latched_depth_point_action_steps",
            "latched_depth_point_distance_first",
            "latched_depth_point_distance_last",
            "latched_depth_point_distance_min",
        ):
            lines.append(f"| `{key}` | {fmt(item.get(key))} |")
        lines.append("")
    if pairs:
        lines.extend(["## Pairwise", "", "| metric | value |", "|---|---:|"])
        for pair in pairs:
            lines.append(f"| `pair` | {pair['pair']} |")
            for key, value in pair.items():
                if key == "pair":
                    continue
                lines.append(f"| `{key}` | {fmt(value)} |")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_trace_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, help="Trace spec as name=path or path.")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, default=None)
    args = parser.parse_args()

    traces = [(name, load_jsonl(path), path) for name, path in map(parse_trace_arg, args.trace)]
    metrics = [trace_metrics(name, rows) for name, rows, _ in traces]
    pairs = []
    if len(traces) >= 2:
        first_name, first_rows, _ = traces[0]
        for name, rows, _ in traces[1:]:
            pairs.append(pair_metrics(first_name, first_rows, name, rows))

    result = {
        "traces": [{"name": name, "path": str(path)} for name, _, path in traces],
        "metrics": metrics,
        "pairs": pairs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.output_md is not None:
        write_markdown(args.output_md, metrics, pairs)
    print(f"[done] wrote {args.output_json}")
    if args.output_md is not None:
        print(f"[done] wrote {args.output_md}")


if __name__ == "__main__":
    main()
