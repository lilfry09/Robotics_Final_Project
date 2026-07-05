#!/usr/bin/env python3
"""Validate a ManiSkill adapter-smoke or replayed-demo HDF5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


POINTCLOUD_REQUIRED = (
    "obs/pointcloud/xyzw",
    "obs/agent/qpos",
    "obs/agent/qvel",
    "actions",
)

STATE_REQUIRED = (
    "obs",
    "actions",
)


def require_dataset(group: h5py.Group, name: str) -> h5py.Dataset:
    if name not in group:
        raise ValueError(f"missing dataset `{group.name}/{name}`")
    obj = group[name]
    if not isinstance(obj, h5py.Dataset):
        raise ValueError(f"`{group.name}/{name}` is not a dataset")
    return obj


def check_numeric(dataset: h5py.Dataset) -> None:
    data = dataset[()]
    if data.dtype.kind in {"f", "i", "u"} and not np.all(np.isfinite(data)):
        raise ValueError(f"non-finite values in `{dataset.name}`")
    if data.shape[0] <= 0:
        raise ValueError(f"empty first dimension in `{dataset.name}`")


def validate(path: Path) -> dict[str, int | str]:
    with h5py.File(path, "r") as h5:
        schema = str(h5.attrs.get("schema", ""))
        obs_mode = str(h5.attrs.get("obs_mode", ""))
        episode_names = sorted(
            name for name in h5.keys() if name.startswith(("episode_", "traj_"))
        )
        if not episode_names:
            raise ValueError("no episode_* or traj_* groups found")

        first_group = h5[episode_names[0]]
        if not isinstance(first_group, h5py.Group):
            raise ValueError(f"`{episode_names[0]}` is not a group")
        if obs_mode != "pointcloud" and "obs/pointcloud/xyzw" in first_group:
            obs_mode = "pointcloud"
        required = POINTCLOUD_REQUIRED if obs_mode == "pointcloud" else STATE_REQUIRED

        for episode_name in episode_names:
            group = h5[episode_name]
            if not isinstance(group, h5py.Group):
                raise ValueError(f"`{episode_name}` is not a group")
            lengths = []
            for dataset_name in required:
                dataset = require_dataset(group, dataset_name)
                check_numeric(dataset)
                lengths.append(dataset.shape[0])
            action_length = lengths[-1]
            obs_lengths = lengths[:-1]
            if any(length not in {action_length, action_length + 1} for length in obs_lengths):
                raise ValueError(f"inconsistent trajectory lengths in `{episode_name}`: {lengths}")
        return {"path": str(path), "schema": schema, "obs_mode": obs_mode, "episodes": len(episode_names)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = validate(args.path)
    print(result)


if __name__ == "__main__":
    main()
