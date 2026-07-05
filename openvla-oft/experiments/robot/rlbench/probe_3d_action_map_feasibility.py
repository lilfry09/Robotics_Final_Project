"""Feasibility probe for Act3D/PerAct-style 3D action maps on RLBench HDF5.

The final DepthVLA-OFT residual/waypoint attempts showed that current policy
heads do not turn depth into robust closed-loop control. Before spending another
7B run on a full 3D action-map head, this script checks a simpler question:

    Do normal RGB-D point candidates geometrically cover the target keypose
    better than cross-sample candidates or an EE-position fallback?

This is not a policy result. It is a cheap GO/NO-GO gate for whether a sampled
3D action map has enough candidate coverage to be worth implementing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class ActionMapSample:
    task: str
    demo: str
    timestep: int
    points: np.ndarray
    ee_pos: np.ndarray
    keypose_xyz: np.ndarray
    next_xyz: np.ndarray
    first_close_xyz: np.ndarray
    pre_first_close_xyz: np.ndarray
    visible_first_close_point_xyz: np.ndarray
    visible_pre_first_close_point_xyz: np.ndarray
    future_xyz: np.ndarray
    final_xyz: np.ndarray
    farthest_future_xyz: np.ndarray


def list_hdf5_files(data_dir: Path) -> list[Path]:
    files = sorted(list(data_dir.glob("*.hdf5")) + list(data_dir.glob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")
    return files


def iter_demo_keys(file_obj: h5py.File) -> list[str]:
    def key_order(key: str) -> int:
        try:
            return int(key.split("_")[-1])
        except ValueError:
            return 10**9

    return sorted(file_obj["data"].keys(), key=key_order)


def infer_task_name(file_path: Path) -> str:
    stem = file_path.stem
    for prefix in ("rlbench_train_", "rlbench_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def regular_indices(total: int, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    if count >= total:
        repeats = int(np.ceil(count / total))
        return np.tile(np.arange(total), repeats)[:count]
    return np.rint(np.linspace(0, total - 1, num=count)).astype(np.int64)


def sample_point_cloud(obs: h5py.Group, timestep: int, points_per_view: int, workspace_radius: float) -> np.ndarray:
    views = []
    ee = np.asarray(obs["ee_pos"][timestep], dtype=np.float32)[:3]
    for key in ("agentview_point_cloud", "eye_in_hand_point_cloud"):
        cloud = np.asarray(obs[key][timestep], dtype=np.float32).reshape(-1, 3)
        valid = np.isfinite(cloud).all(axis=1)
        if workspace_radius > 0:
            valid = valid & (np.linalg.norm(cloud - ee[None, :], axis=1) <= workspace_radius)
        cloud = cloud[valid]
        if cloud.size == 0:
            cloud = ee.reshape(1, 3)
        idx = regular_indices(cloud.shape[0], points_per_view)
        views.append(cloud[idx])
    return np.concatenate(views, axis=0).astype(np.float32)


def load_samples(
    data_dir: Path,
    max_samples: int | None,
    stride: int,
    points_per_view: int,
    workspace_radius: float,
    future_horizon: int,
) -> list[ActionMapSample]:
    samples: list[ActionMapSample] = []
    for file_path in list_hdf5_files(data_dir):
        task = infer_task_name(file_path)
        with h5py.File(file_path, "r") as f:
            for demo_key in iter_demo_keys(f):
                demo = f["data"][demo_key]
                obs = demo["obs"]
                required = (
                    "agentview_point_cloud",
                    "eye_in_hand_point_cloud",
                    "ee_pos",
                    "proprio",
                    "rlbench_abs_gripper_pose",
                    "rlbench_next_abs_gripper_pose",
                )
                missing = [key for key in required if key not in obs]
                if missing:
                    raise KeyError(f"{file_path}:{demo_key}/obs missing {missing}")
                if "rlbench_keypose_action" not in demo:
                    raise KeyError(f"{file_path}:{demo_key} missing rlbench_keypose_action")
                length = int(demo["rlbench_keypose_action"].shape[0])
                abs_xyz = np.asarray(obs["rlbench_abs_gripper_pose"][:, :3], dtype=np.float32)
                gripper_open = np.asarray(obs["proprio"][:, -1], dtype=np.float32)
                close_indices = np.flatnonzero(gripper_open < 0.5)
                first_close_index = int(close_indices[0]) if close_indices.size else abs_xyz.shape[0] - 1
                pre_first_close_index = max(0, first_close_index - max(1, int(future_horizon)))
                first_close_xyz = abs_xyz[first_close_index].astype(np.float32)
                pre_first_close_xyz = abs_xyz[pre_first_close_index].astype(np.float32)
                for timestep in range(0, length, stride):
                    keypose = np.asarray(demo["rlbench_keypose_action"][timestep], dtype=np.float32)
                    next_pose = np.asarray(obs["rlbench_next_abs_gripper_pose"][timestep], dtype=np.float32)
                    ee_pos = np.asarray(obs["ee_pos"][timestep], dtype=np.float32)[:3]
                    if not np.isfinite(keypose[:3]).all() or not np.isfinite(next_pose[:3]).all():
                        continue
                    future_index = min(timestep + max(1, int(future_horizon)), length - 1)
                    future_xyz = abs_xyz[future_index]
                    final_xyz = abs_xyz[-1]
                    future_window = abs_xyz[timestep:]
                    farthest_index = int(np.argmax(np.linalg.norm(future_window - ee_pos.reshape(1, 3), axis=1)))
                    farthest_future_xyz = future_window[farthest_index]
                    points = sample_point_cloud(obs, timestep, points_per_view, workspace_radius)
                    visible_first_close_point_xyz = nearest_point(points, first_close_xyz)
                    visible_pre_first_close_point_xyz = nearest_point(points, pre_first_close_xyz)
                    samples.append(
                        ActionMapSample(
                            task=task,
                            demo=demo_key,
                            timestep=timestep,
                            points=points,
                            ee_pos=ee_pos,
                            keypose_xyz=keypose[:3].astype(np.float32),
                            next_xyz=next_pose[:3].astype(np.float32),
                            first_close_xyz=first_close_xyz,
                            pre_first_close_xyz=pre_first_close_xyz,
                            visible_first_close_point_xyz=visible_first_close_point_xyz,
                            visible_pre_first_close_point_xyz=visible_pre_first_close_point_xyz,
                            future_xyz=future_xyz.astype(np.float32),
                            final_xyz=final_xyz.astype(np.float32),
                            farthest_future_xyz=farthest_future_xyz.astype(np.float32),
                        )
                    )
                    if max_samples is not None and len(samples) >= max_samples:
                        return samples
    if not samples:
        raise RuntimeError(f"No valid samples found in {data_dir}")
    return samples


def nearest_distance(points: np.ndarray, target: np.ndarray) -> float:
    if points.size == 0:
        return float("inf")
    dist = np.linalg.norm(points - target.reshape(1, 3), axis=1)
    return float(np.min(dist))


def nearest_point(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.asarray(target, dtype=np.float32).reshape(3)
    dist = np.linalg.norm(points - target.reshape(1, 3), axis=1)
    return points[int(np.argmin(dist))].astype(np.float32)


def summarize(errors: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    out = {
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
    }
    for threshold in thresholds:
        out[f"within_{threshold:.3f}m"] = float(np.mean(errors <= threshold))
    return out


def evaluate(samples: list[ActionMapSample], target_name: str, thresholds: list[float], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(samples))
    if len(samples) > 1:
        # A deterministic derangement-like corruption: roll by a random nonzero offset.
        offset = int(rng.integers(1, len(samples)))
        cross_order = np.roll(order, offset)
    else:
        cross_order = order

    normal_errors = []
    cross_errors = []
    ee_errors = []
    for idx, sample in enumerate(samples):
        if target_name == "keypose":
            target = sample.keypose_xyz
        elif target_name == "next_pose":
            target = sample.next_xyz
        elif target_name == "first_close_pose":
            target = sample.first_close_xyz
        elif target_name == "pre_first_close_pose":
            target = sample.pre_first_close_xyz
        elif target_name == "visible_first_close_point":
            target = sample.visible_first_close_point_xyz
        elif target_name == "visible_pre_first_close_point":
            target = sample.visible_pre_first_close_point_xyz
        elif target_name == "future_pose":
            target = sample.future_xyz
        elif target_name == "final_pose":
            target = sample.final_xyz
        elif target_name == "farthest_future_pose":
            target = sample.farthest_future_xyz
        else:
            raise ValueError(f"Unknown target_name: {target_name}")
        normal_errors.append(nearest_distance(sample.points, target))
        cross_errors.append(nearest_distance(samples[int(cross_order[idx])].points, target))
        ee_errors.append(float(np.linalg.norm(sample.ee_pos - target)))

    normal = np.asarray(normal_errors, dtype=np.float64)
    cross = np.asarray(cross_errors, dtype=np.float64)
    ee = np.asarray(ee_errors, dtype=np.float64)
    result = {
        "normal": summarize(normal, thresholds),
        "cross_sample": summarize(cross, thresholds),
        "ee_fallback": summarize(ee, thresholds),
        "advantage_over_cross_median_m": float(np.median(cross) - np.median(normal)),
        "advantage_over_ee_median_m": float(np.median(ee) - np.median(normal)),
        "paired_normal_vs_cross_l1_m": float(np.mean(np.abs(normal - cross))),
    }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_samples(
        data_dir=Path(args.data_dir),
        max_samples=args.max_samples,
        stride=args.stride,
        points_per_view=args.points_per_view,
        workspace_radius=args.workspace_radius,
        future_horizon=args.future_horizon,
    )
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    result = evaluate(samples, target_name=args.target, thresholds=thresholds, seed=args.seed)
    tasks: dict[str, int] = {}
    for sample in samples:
        tasks[sample.task] = tasks.get(sample.task, 0) + 1
    return {
        "config": {
            "data_dir": str(args.data_dir),
            "target": args.target,
            "max_samples": args.max_samples,
            "stride": args.stride,
            "points_per_view": args.points_per_view,
            "workspace_radius": args.workspace_radius,
            "future_horizon": args.future_horizon,
            "thresholds": thresholds,
            "seed": args.seed,
        },
        "num_samples": len(samples),
        "tasks": tasks,
        "result": result,
    }


def print_result(payload: dict[str, Any]) -> None:
    result = payload["result"]
    print("=" * 80)
    print("RLBench 3D Action-Map Candidate Feasibility Probe")
    print("=" * 80)
    print(f"samples: {payload['num_samples']}")
    print(f"tasks: {payload['tasks']}")
    for mode in ("normal", "cross_sample", "ee_fallback"):
        row = result[mode]
        print(
            f"{mode:13s} median={row['median_m']:.4f}m "
            f"mean={row['mean_m']:.4f}m p90={row['p90_m']:.4f}m"
        )
    print(f"advantage over cross median: {result['advantage_over_cross_median_m']:+.4f}m")
    print(f"advantage over EE median:    {result['advantage_over_ee_median_m']:+.4f}m")
    print(f"paired normal-vs-cross L1:   {result['paired_normal_vs_cross_l1_m']:.4f}m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--target",
        choices=(
            "keypose",
            "next_pose",
            "first_close_pose",
            "pre_first_close_pose",
            "visible_first_close_point",
            "visible_pre_first_close_point",
            "future_pose",
            "final_pose",
            "farthest_future_pose",
        ),
        default="keypose",
    )
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--points_per_view", type=int, default=1024)
    parser.add_argument("--workspace_radius", type=float, default=0.0)
    parser.add_argument("--future_horizon", type=int, default=10)
    parser.add_argument("--thresholds", default="0.01,0.02,0.05,0.10")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    payload = run(args)
    print_result(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
