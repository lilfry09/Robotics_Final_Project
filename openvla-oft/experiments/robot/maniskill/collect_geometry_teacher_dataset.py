#!/usr/bin/env python3
"""Collect object-feature imitation data from the PickCube geometry controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval_pointcloud_action_decoder import clip_action, extract_proprio, require_maniskill, scalar_bool, squeeze_batch, to_numpy
from eval_pointcloud_geometry_controller import ControllerState, cube_center_from_obs, step_controller
from train_object_feature_action_decoder import make_feature


PHASE_NAMES = ("approach", "descend", "close", "lift", "move_goal", "hold_goal", "no_cube")


def phase_one_hot(phase: str) -> np.ndarray:
    values = np.zeros(len(PHASE_NAMES), dtype=np.float32)
    if phase in PHASE_NAMES:
        values[PHASE_NAMES.index(phase)] = 1.0
    return values


def extract_goal(obs: dict[str, Any]) -> np.ndarray:
    return squeeze_batch(obs["extra"]["goal_pos"]).reshape(-1)[:3].astype(np.float32)


def extract_tcp(obs: dict[str, Any]) -> np.ndarray:
    return squeeze_batch(obs["extra"]["tcp_pose"]).reshape(-1)[:3].astype(np.float32)


def extract_grasped(obs: dict[str, Any]) -> float:
    return float(scalar_bool(obs["extra"]["is_grasped"]))


def collect_episode(
    env: Any,
    seed: int,
    cube_seg_id: int,
    max_steps: int,
    gain: float,
    max_xyz_action: float,
    include_phase: bool,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    state = ControllerState()
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    phases: list[str] = []
    rewards: list[float] = []
    successes: list[bool] = []
    terminated = False
    truncated = False
    for _ in range(max_steps):
        phase_before = state.phase
        cube_center = cube_center_from_obs(obs, cube_seg_id)
        if cube_center is None:
            feature_cube = np.zeros(3, dtype=np.float32)
            cube_valid = 0.0
        else:
            feature_cube = cube_center.astype(np.float32)
            cube_valid = 1.0
        feature = make_feature(
            feature_cube,
            cube_valid,
            extract_tcp(obs),
            extract_goal(obs),
            extract_grasped(obs),
            extract_proprio(obs),
        )
        if include_phase:
            feature = np.concatenate([feature, phase_one_hot(phase_before)], axis=0).astype(np.float32)
        action, debug = step_controller(obs, state, cube_center, gain, max_xyz_action)
        action = clip_action(action, env.action_space)
        obs, reward, terminated, truncated, info = env.step(action)
        features.append(feature)
        actions.append(action.astype(np.float32))
        phases.append(phase_before)
        rewards.append(float(to_numpy(reward).reshape(-1)[0]))
        if isinstance(info, dict) and "success" in info:
            successes.append(scalar_bool(info["success"]))
        if scalar_bool(terminated) or scalar_bool(truncated):
            break
    return {
        "features": np.stack(features, axis=0).astype(np.float32) if features else np.zeros((0, 0), dtype=np.float32),
        "actions": np.stack(actions, axis=0).astype(np.float32) if actions else np.zeros((0, 0), dtype=np.float32),
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
    all_features: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
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
                args.max_steps,
                args.gain,
                args.max_xyz_action,
                args.include_phase,
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
                all_features.append(result["features"])
                all_actions.append(result["actions"])
                all_episode_ids.append(np.full(result["length"], accepted, dtype=np.int64))
                accepted += 1
            if args.log_every and (attempts + 1) % args.log_every == 0:
                print(f"attempts={attempts + 1} accepted={accepted} last_success={result['success']}")
            attempts += 1
    finally:
        env.close()

    if not all_features:
        raise RuntimeError("no episodes collected")
    features = np.concatenate(all_features, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    episode_ids = np.concatenate(all_episode_ids, axis=0).astype(np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        actions=actions,
        episode_ids=episode_ids,
        cube_seg_id=np.asarray([args.cube_seg_id], dtype=np.int64),
        include_phase=np.asarray([int(args.include_phase)], dtype=np.int64),
        phase_names=np.asarray(PHASE_NAMES),
    )
    summary = {
        "output": str(args.output),
        "env_id": args.env_id,
        "control_mode": args.control_mode,
        "attempts": attempts,
        "accepted_episodes": accepted,
        "samples": int(len(actions)),
        "success_only": bool(args.success_only),
        "include_phase": bool(args.include_phase),
        "phase_names": list(PHASE_NAMES),
        "episodes": episodes,
        "note": "Object-feature/action data generated by the pointcloud geometry controller.",
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
    parser.add_argument("--seed", type=int, default=5200)
    parser.add_argument("--cube_seg_id", type=int, default=18)
    parser.add_argument("--gain", type=float, default=5.0)
    parser.add_argument("--max_xyz_action", type=float, default=0.25)
    parser.add_argument("--success_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_phase", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
