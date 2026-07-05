#!/usr/bin/env python3
"""Collect raw pointcloud teacher data from the PickCube geometry controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from collect_geometry_teacher_dataset import PHASE_NAMES, extract_goal, extract_grasped, extract_tcp
from eval_pointcloud_action_decoder import clip_action, extract_proprio, require_maniskill, scalar_bool, to_numpy
from eval_pointcloud_geometry_controller import ControllerState, cube_center_from_obs, step_controller
from train_pointcloud_action_decoder import select_points


TASK_FEATURE_NAMES = (
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "goal_x",
    "goal_y",
    "goal_z",
    "is_grasped",
    "tcp_to_goal_x",
    "tcp_to_goal_y",
    "tcp_to_goal_z",
    "tcp_to_goal_dist",
)


def phase_index(phase: str) -> int:
    return PHASE_NAMES.index(phase) if phase in PHASE_NAMES else PHASE_NAMES.index("no_cube")


def make_task_feature(obs: dict[str, Any]) -> np.ndarray:
    tcp = extract_tcp(obs)
    goal = extract_goal(obs)
    tcp_to_goal = goal - tcp
    dist = float(np.linalg.norm(tcp_to_goal))
    return np.concatenate(
        [
            tcp,
            goal,
            np.asarray([extract_grasped(obs)], dtype=np.float32),
            tcp_to_goal.astype(np.float32),
            np.asarray([dist], dtype=np.float32),
            extract_proprio(obs),
        ],
        axis=0,
    ).astype(np.float32)


def mask_low_points(xyzw: np.ndarray, min_z: float) -> np.ndarray:
    if min_z <= -1e8:
        return xyzw
    filtered = np.asarray(xyzw).copy()
    valid = filtered[..., 2] >= min_z
    if filtered.shape[-1] >= 4:
        filtered[..., 3] = np.where(valid, filtered[..., 3], 0.0)
    return filtered


def extract_points(obs: dict[str, Any], num_points: int, rng: np.random.Generator, min_z: float = -1e9) -> np.ndarray:
    pointcloud = obs["pointcloud"]
    xyzw = to_numpy(pointcloud["xyzw"])
    rgb = to_numpy(pointcloud["rgb"]) if "rgb" in pointcloud else None
    if xyzw.shape[:1] == (1,):
        xyzw = xyzw[0]
    if rgb is not None and rgb.shape[:1] == (1,):
        rgb = rgb[0]
    xyzw = mask_low_points(xyzw, min_z)
    return select_points(xyzw, rgb, num_points, rng)


def collect_episode(
    env: Any,
    seed: int,
    cube_seg_id: int,
    num_points: int,
    max_steps: int,
    gain: float,
    max_xyz_action: float,
    rng: np.random.Generator,
    min_z: float,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    state = ControllerState()
    points: list[np.ndarray] = []
    cube_centers: list[np.ndarray] = []
    cube_valid: list[float] = []
    task_features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    phase_labels: list[int] = []
    phases: list[str] = []
    rewards: list[float] = []
    successes: list[bool] = []
    terminated = False
    truncated = False
    for _ in range(max_steps):
        phase_before = state.phase
        cube_center = cube_center_from_obs(obs, cube_seg_id)
        if cube_center is None:
            cube_centers.append(np.zeros(3, dtype=np.float32))
            cube_valid.append(0.0)
        else:
            cube_centers.append(cube_center.astype(np.float32))
            cube_valid.append(1.0)
        points.append(extract_points(obs, num_points, rng, min_z))
        task_features.append(make_task_feature(obs))
        phase_labels.append(phase_index(phase_before))
        phases.append(phase_before)
        action, _ = step_controller(obs, state, cube_center, gain, max_xyz_action)
        action = clip_action(action, env.action_space)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action.astype(np.float32))
        rewards.append(float(to_numpy(reward).reshape(-1)[0]))
        if isinstance(info, dict) and "success" in info:
            successes.append(scalar_bool(info["success"]))
        if scalar_bool(terminated) or scalar_bool(truncated):
            break

    return {
        "points": np.stack(points, axis=0).astype(np.float32) if points else np.zeros((0, num_points, 6), dtype=np.float32),
        "cube_centers": np.stack(cube_centers, axis=0).astype(np.float32)
        if cube_centers
        else np.zeros((0, 3), dtype=np.float32),
        "cube_valid": np.asarray(cube_valid, dtype=np.float32),
        "task_features": np.stack(task_features, axis=0).astype(np.float32)
        if task_features
        else np.zeros((0, 0), dtype=np.float32),
        "actions": np.stack(actions, axis=0).astype(np.float32) if actions else np.zeros((0, 0), dtype=np.float32),
        "phase_labels": np.asarray(phase_labels, dtype=np.int64),
        "phases": phases,
        "success": bool(successes[-1]) if successes else False,
        "length": len(actions),
        "reward_sum": float(np.sum(rewards)),
        "terminated": scalar_bool(terminated),
        "truncated": scalar_bool(truncated),
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    gym = require_maniskill()
    env = gym.make(args.env_id, obs_mode="pointcloud", control_mode=args.control_mode, max_episode_steps=args.max_steps)
    rng = np.random.default_rng(args.seed + 1009)
    all_points: list[np.ndarray] = []
    all_cube_centers: list[np.ndarray] = []
    all_cube_valid: list[np.ndarray] = []
    all_task_features: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_phase_labels: list[np.ndarray] = []
    all_episode_ids: list[np.ndarray] = []
    episodes: list[dict[str, Any]] = []
    accepted = 0
    attempts = 0
    try:
        while accepted < args.target_successes and attempts < args.max_attempts:
            seed = args.seed + attempts
            result = collect_episode(
                env,
                seed,
                args.cube_seg_id,
                args.num_points,
                args.max_steps,
                args.gain,
                args.max_xyz_action,
                rng,
                args.min_z,
            )
            record = {
                "attempt": attempts,
                "seed": seed,
                "success": result["success"],
                "length": result["length"],
                "reward_sum": result["reward_sum"],
                "terminated": result["terminated"],
                "truncated": result["truncated"],
                "phase_counts": {phase: result["phases"].count(phase) for phase in sorted(set(result["phases"]))},
            }
            episodes.append(record)
            if result["success"] or not args.success_only:
                all_points.append(result["points"])
                all_cube_centers.append(result["cube_centers"])
                all_cube_valid.append(result["cube_valid"])
                all_task_features.append(result["task_features"])
                all_actions.append(result["actions"])
                all_phase_labels.append(result["phase_labels"])
                all_episode_ids.append(np.full(result["length"], accepted, dtype=np.int64))
                accepted += 1
            attempts += 1
            if args.log_every and attempts % args.log_every == 0:
                print(f"attempts={attempts} accepted={accepted} last_success={result['success']}")
    finally:
        env.close()

    if not all_actions:
        raise RuntimeError("no episodes collected")
    points = np.concatenate(all_points, axis=0).astype(np.float32)
    cube_centers = np.concatenate(all_cube_centers, axis=0).astype(np.float32)
    cube_valid = np.concatenate(all_cube_valid, axis=0).astype(np.float32)
    task_features = np.concatenate(all_task_features, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    phase_labels = np.concatenate(all_phase_labels, axis=0).astype(np.int64)
    episode_ids = np.concatenate(all_episode_ids, axis=0).astype(np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points=points,
        cube_centers=cube_centers,
        cube_valid=cube_valid,
        task_features=task_features,
        actions=actions,
        phase_labels=phase_labels,
        episode_ids=episode_ids,
        phase_names=np.asarray(PHASE_NAMES),
        task_feature_names=np.asarray(list(TASK_FEATURE_NAMES) + [f"proprio_{idx}" for idx in range(task_features.shape[-1] - len(TASK_FEATURE_NAMES))]),
        cube_seg_id=np.asarray([args.cube_seg_id], dtype=np.int64),
        num_points=np.asarray([args.num_points], dtype=np.int64),
        min_z=np.asarray([args.min_z], dtype=np.float32),
    )
    summary = {
        "output": str(args.output),
        "env_id": args.env_id,
        "control_mode": args.control_mode,
        "attempts": attempts,
        "accepted_episodes": accepted,
        "samples": int(len(actions)),
        "num_points": int(args.num_points),
        "min_z": float(args.min_z),
        "success_only": bool(args.success_only),
        "phase_names": list(PHASE_NAMES),
        "episodes": episodes,
        "note": "Raw pointcloud/task/action data generated by the PickCube geometry controller.",
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.summary}")
    print(json.dumps({k: summary[k] for k in ("attempts", "accepted_episodes", "samples")}, indent=2))
    print(f"wrote {args.output}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--env_id", default="PickCube-v1")
    parser.add_argument("--control_mode", default="pd_ee_delta_pos")
    parser.add_argument("--target_successes", type=int, default=30)
    parser.add_argument("--max_attempts", type=int, default=60)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--min_z", type=float, default=-1e9)
    parser.add_argument("--seed", type=int, default=5400)
    parser.add_argument("--cube_seg_id", type=int, default=18)
    parser.add_argument("--gain", type=float, default=5.0)
    parser.add_argument("--max_xyz_action", type=float, default=0.25)
    parser.add_argument("--success_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
