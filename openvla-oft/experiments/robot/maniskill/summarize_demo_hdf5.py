#!/usr/bin/env python3
"""Print a concise summary of a ManiSkill trajectory HDF5 file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def natural_key(name: str) -> tuple[str, int | str]:
    prefix, _, suffix = name.rpartition("_")
    if suffix.isdigit():
        return prefix, int(suffix)
    return name, suffix


def dataset_shape(group: h5py.Group, key: str) -> list[int] | None:
    if key not in group or not isinstance(group[key], h5py.Dataset):
        return None
    return list(group[key].shape)


def collect_env_state_keys(group: h5py.Group) -> list[str]:
    if "env_states" not in group or not isinstance(group["env_states"], h5py.Group):
        return []
    keys: list[str] = []

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            keys.append(name)

    group["env_states"].visititems(visit)
    return sorted(keys)[:40]


def summarize(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        traj_names = sorted(
            [name for name in h5.keys() if name.startswith(("traj_", "episode_"))],
            key=natural_key,
        )
        if not traj_names:
            raise ValueError(f"no traj_* or episode_* groups in {path}")

        lengths: list[int] = []
        action_dims: list[int] = []
        final_success: list[bool] = []
        first_shapes: dict[str, list[int] | None] = {}
        env_state_keys: list[str] = []

        for idx, name in enumerate(traj_names):
            group = h5[name]
            if not isinstance(group, h5py.Group):
                continue
            if "actions" in group:
                actions = group["actions"]
                lengths.append(int(actions.shape[0]))
                action_dims.append(int(actions.shape[-1]) if actions.ndim > 1 else 1)
            if "success" in group:
                success_arr = np.asarray(group["success"][()])
                if success_arr.size:
                    final_success.append(bool(success_arr.reshape(-1)[-1]))
            if idx == 0:
                for key in (
                    "actions",
                    "obs/pointcloud/xyzw",
                    "obs/pointcloud/rgb",
                    "obs/pointcloud/segmentation",
                    "obs/agent/qpos",
                    "obs/agent/qvel",
                    "obs/sensor_param/base_camera/intrinsic_cv",
                    "obs/sensor_param/base_camera/extrinsic_cv",
                ):
                    first_shapes[key] = dataset_shape(group, key)
                env_state_keys = collect_env_state_keys(group)

        length_arr = np.asarray(lengths, dtype=np.float32)
        return {
            "path": str(path),
            "num_trajectories": len(traj_names),
            "length_min": int(length_arr.min()) if length_arr.size else None,
            "length_mean": float(length_arr.mean()) if length_arr.size else None,
            "length_max": int(length_arr.max()) if length_arr.size else None,
            "action_dims": sorted(set(action_dims)),
            "final_success_rate": float(np.mean(final_success)) if final_success else None,
            "first_trajectory_shapes": first_shapes,
            "env_state_keys_first_trajectory": env_state_keys,
            "attrs": {key: str(value) for key, value in h5.attrs.items()},
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = summarize(args.path)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"path: {summary['path']}")
        print(f"trajectories: {summary['num_trajectories']}")
        print(
            "length min/mean/max: "
            f"{summary['length_min']} / {summary['length_mean']:.2f} / {summary['length_max']}"
        )
        print(f"action dims: {summary['action_dims']}")
        print(f"final success rate: {summary['final_success_rate']}")
        print("first trajectory shapes:")
        for key, shape in summary["first_trajectory_shapes"].items():
            print(f"  {key}: {shape}")
        if summary["env_state_keys_first_trajectory"]:
            print("env state keys, first trajectory:")
            for key in summary["env_state_keys_first_trajectory"]:
                print(f"  {key}")


if __name__ == "__main__":
    main()
