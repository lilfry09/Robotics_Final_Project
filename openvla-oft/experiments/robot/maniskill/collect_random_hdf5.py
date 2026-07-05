#!/usr/bin/env python3
"""Collect a tiny ManiSkill random-policy HDF5 smoke dataset.

This is not the final expert-demonstration pipeline. It is an adapter smoke
that verifies observation/action extraction, HDF5 schema, and normal/null/cross
diagnostic plumbing before spending time on demonstrations or policy training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def require_maniskill() -> tuple[Any, Any]:
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment diagnostic
        raise SystemExit(
            "ManiSkill3 is not available. Run "
            "`experiments/robot/maniskill/setup_maniskill_env.sh` first. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return gym, mani_skill.envs


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    else:
        value = np.asarray(value)
    return np.asarray(value)


def squeeze_batch(value: Any) -> np.ndarray:
    arr = to_numpy(value)
    if arr.shape[:1] == (1,):
        arr = arr[0]
    return arr


def flatten_numeric(prefix: str, value: Any, out: dict[str, np.ndarray]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_numeric(f"{prefix}/{key}" if prefix else str(key), child, out)
        return
    arr = squeeze_batch(value)
    if arr.dtype.kind in {"b", "i", "u", "f"}:
        out[prefix] = arr


def collect_episode(env: Any, max_steps: int) -> dict[str, list[np.ndarray]]:
    obs, _ = env.reset()
    episode: dict[str, list[np.ndarray]] = {}
    for step_idx in range(max_steps):
        flat_obs: dict[str, np.ndarray] = {}
        flatten_numeric("obs", obs, flat_obs)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        for key, value in flat_obs.items():
            episode.setdefault(key, []).append(value)
        episode.setdefault("actions", []).append(squeeze_batch(action))
        episode.setdefault("rewards", []).append(np.asarray(reward, dtype=np.float32).reshape(()))
        episode.setdefault("terminated", []).append(np.asarray(terminated, dtype=np.bool_).reshape(()))
        episode.setdefault("truncated", []).append(np.asarray(truncated, dtype=np.bool_).reshape(()))

        success = False
        if isinstance(info, dict) and "success" in info:
            success_arr = squeeze_batch(info["success"])
            success = bool(np.asarray(success_arr).reshape(-1)[0])
        episode.setdefault("success", []).append(np.asarray(success, dtype=np.bool_).reshape(()))

        obs = next_obs
        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
            break
        if step_idx + 1 >= max_steps:
            break
    return episode


def write_episode(group: h5py.Group, episode: dict[str, list[np.ndarray]]) -> None:
    for key, values in episode.items():
        try:
            stacked = np.stack(values, axis=0)
        except ValueError:
            # Keep ragged point clouds explicit instead of silently writing bad data.
            ragged = group.create_group(key)
            for idx, value in enumerate(values):
                ragged.create_dataset(str(idx), data=value, compression="gzip")
            continue
        group.create_dataset(key, data=stacked, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="PushCube-v1")
    parser.add_argument("--obs_mode", default="pointcloud", choices=["state", "rgbd", "pointcloud"])
    parser.add_argument("--control_mode", default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("experiments/logs/maniskill_random_smoke.hdf5"))
    parser.add_argument("--language", default=None)
    args = parser.parse_args()

    gym, _ = require_maniskill()
    env_kwargs: dict[str, Any] = {"obs_mode": args.obs_mode}
    if args.control_mode:
        env_kwargs["control_mode"] = args.control_mode
    env = gym.make(args.env_id, **env_kwargs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as h5:
        h5.attrs["env_id"] = args.env_id
        h5.attrs["obs_mode"] = args.obs_mode
        h5.attrs["control_mode"] = args.control_mode or ""
        h5.attrs["language"] = args.language or args.env_id
        h5.attrs["schema"] = "maniskill_random_smoke_v1"
        h5.attrs["note"] = "Random-policy adapter smoke, not expert demonstrations."
        for episode_idx in range(args.episodes):
            episode = collect_episode(env, args.max_steps)
            write_episode(h5.create_group(f"episode_{episode_idx:04d}"), episode)
    env.close()

    summary = {
        "output": str(args.output),
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
