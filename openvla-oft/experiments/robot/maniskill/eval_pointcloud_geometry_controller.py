#!/usr/bin/env python3
"""Point-cloud geometry controller smoke for ManiSkill PickCube.

This is a feasibility probe, not a learned VLA policy. It uses point-cloud
segmentation to estimate the cube position and uses the task-provided goal/tcp
proprio fields from the observation. The goal is to check whether the task is
geometrically solvable from RGB-D/pointcloud style observations before investing
in a heavier learned temporal policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from eval_pointcloud_action_decoder import require_maniskill, scalar_bool, squeeze_batch, to_numpy
from train_pointcloud_action_decoder import natural_key


@dataclass
class ControllerState:
    phase: str = "approach"
    phase_steps: int = 0
    last_cube: np.ndarray | None = None


def cube_center_from_obs(obs: dict[str, Any], cube_seg_id: int) -> np.ndarray | None:
    pointcloud = obs["pointcloud"]
    xyz = squeeze_batch(pointcloud["xyzw"])[..., :3]
    seg = squeeze_batch(pointcloud["segmentation"]).reshape(-1)
    pts = xyz[seg == cube_seg_id]
    pts = pts[np.all(np.isfinite(pts), axis=-1)]
    if len(pts) < 3:
        return None
    return pts.mean(axis=0).astype(np.float32)


def load_cross_cube_centers(path: Path, cube_seg_id: int, max_centers: int, seed: int) -> np.ndarray:
    centers: list[np.ndarray] = []
    with h5py.File(path, "r") as h5:
        names = sorted([name for name in h5.keys() if name.startswith(("traj_", "episode_"))], key=natural_key)
        for name in names:
            group = h5[name]
            if not isinstance(group, h5py.Group):
                continue
            xyzw = group["obs/pointcloud/xyzw"]
            seg = group["obs/pointcloud/segmentation"]
            length = min(len(xyzw), len(seg))
            for idx in range(length):
                xyz_i = np.asarray(xyzw[idx])[..., :3]
                seg_i = np.asarray(seg[idx]).reshape(-1)
                pts = xyz_i[seg_i == cube_seg_id]
                pts = pts[np.all(np.isfinite(pts), axis=-1)]
                if len(pts) >= 3:
                    centers.append(pts.mean(axis=0).astype(np.float32))
                if len(centers) >= max_centers:
                    break
            if len(centers) >= max_centers:
                break
    if not centers:
        raise ValueError(f"no cube centers found in {path}")
    rng = np.random.default_rng(seed)
    centers_arr = np.stack(centers, axis=0)
    rng.shuffle(centers_arr)
    return centers_arr


def get_goal(obs: dict[str, Any]) -> np.ndarray:
    return squeeze_batch(obs["extra"]["goal_pos"]).reshape(-1)[:3].astype(np.float32)


def get_tcp(obs: dict[str, Any]) -> np.ndarray:
    return squeeze_batch(obs["extra"]["tcp_pose"]).reshape(-1)[:3].astype(np.float32)


def get_grasped(obs: dict[str, Any]) -> bool:
    return scalar_bool(obs["extra"]["is_grasped"])


def step_controller(
    obs: dict[str, Any],
    state: ControllerState,
    cube_center: np.ndarray | None,
    gain: float,
    max_xyz_action: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    tcp = get_tcp(obs)
    goal = get_goal(obs)
    grasped = get_grasped(obs)
    if cube_center is not None:
        state.last_cube = cube_center.astype(np.float32)
    elif state.last_cube is not None and state.phase not in {"approach", "hold_goal"}:
        cube_center = state.last_cube

    if cube_center is None:
        target = tcp.copy()
        gripper = 1.0
        action_xyz = np.zeros(3, dtype=np.float32)
        return np.asarray([0.0, 0.0, 0.0, gripper], dtype=np.float32), {
            "phase": "no_cube",
            "target": target.tolist(),
            "tcp": tcp.tolist(),
            "goal": goal.tolist(),
            "grasped": grasped,
        }

    cube = cube_center.astype(np.float32)
    state.phase_steps += 1
    xy_err = float(np.linalg.norm(tcp[:2] - cube[:2]))
    # The visible cube point-cloud mean is biased toward the top/front surface.
    # In ManiSkill PickCube demos, the Panda TCP grasps around z ~= 0.02m, so
    # target slightly below the observed point mean rather than above it.
    grasp_z = max(float(cube[2] - 0.012), 0.018)

    if state.phase == "approach":
        target = np.asarray([cube[0], cube[1], max(cube[2] + 0.12, 0.12)], dtype=np.float32)
        gripper = 1.0
        if xy_err < 0.025 and abs(float(tcp[2] - target[2])) < 0.025:
            state.phase = "descend"
            state.phase_steps = 0
    elif state.phase == "descend":
        target = np.asarray([cube[0], cube[1], grasp_z], dtype=np.float32)
        gripper = 1.0
        if np.linalg.norm(tcp - target) < 0.025 or state.phase_steps > 30:
            state.phase = "close"
            state.phase_steps = 0
    elif state.phase == "close":
        target = np.asarray([cube[0], cube[1], grasp_z], dtype=np.float32)
        gripper = -1.0
        if grasped:
            state.phase = "lift"
            state.phase_steps = 0
        elif state.phase_steps > 20:
            state.phase = "descend"
            state.phase_steps = 0
    elif state.phase == "lift":
        target = np.asarray([tcp[0], tcp[1], max(goal[2] + 0.08, cube[2] + 0.18)], dtype=np.float32)
        gripper = -1.0
        if not grasped and state.phase_steps > 5:
            state.phase = "descend"
            state.phase_steps = 0
        elif tcp[2] > target[2] - 0.025 or state.phase_steps > 35:
            state.phase = "move_goal"
            state.phase_steps = 0
    elif state.phase == "move_goal":
        target = goal.astype(np.float32)
        gripper = -1.0
        if not grasped:
            state.phase = "approach"
            state.phase_steps = 0
        elif np.linalg.norm(tcp[:2] - goal[:2]) < 0.03 and abs(float(tcp[2] - target[2])) < 0.04:
            state.phase = "hold_goal"
            state.phase_steps = 0
    else:
        target = goal.astype(np.float32)
        gripper = -1.0
        if not grasped:
            state.phase = "approach"
            state.phase_steps = 0

    action_xyz = np.clip((target - tcp) * gain, -max_xyz_action, max_xyz_action)
    action = np.concatenate([action_xyz, np.asarray([gripper], dtype=np.float32)]).astype(np.float32)
    return action, {
        "phase": state.phase,
        "target": target.tolist(),
        "tcp": tcp.tolist(),
        "cube": cube.tolist(),
        "goal": goal.tolist(),
        "grasped": grasped,
        "xy_err": xy_err,
    }


def select_cube_center(
    obs: dict[str, Any],
    point_mode: str,
    cube_seg_id: int,
    cross_centers: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if point_mode == "normal":
        return cube_center_from_obs(obs, cube_seg_id)
    if point_mode == "null":
        return None
    if point_mode == "cross_demo":
        if cross_centers is None:
            raise ValueError("cross_demo requires --cross_hdf5")
        return cross_centers[int(rng.integers(0, len(cross_centers)))]
    raise ValueError(f"unknown point mode: {point_mode}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gym = require_maniskill()
    rng = np.random.default_rng(args.seed)
    cross_centers = None
    if args.cross_hdf5 is not None:
        cross_centers = load_cross_cube_centers(args.cross_hdf5, args.cube_seg_id, args.cross_samples, args.seed + 97)

    env = gym.make(
        args.env_id,
        obs_mode="pointcloud",
        control_mode=args.control_mode,
        max_episode_steps=args.max_steps,
    )
    episodes: list[dict[str, Any]] = []
    for episode_idx in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode_idx)
        state = ControllerState()
        rewards: list[float] = []
        phases: list[str] = []
        successes: list[bool] = []
        final_debug: dict[str, Any] = {}
        terminated = False
        truncated = False
        for _ in range(args.max_steps):
            cube_center = select_cube_center(obs, args.point_mode, args.cube_seg_id, cross_centers, rng)
            action, debug = step_controller(obs, state, cube_center, args.gain, args.max_xyz_action)
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(to_numpy(reward).reshape(-1)[0]))
            phases.append(str(debug["phase"]))
            final_debug = debug
            if isinstance(info, dict) and "success" in info:
                successes.append(scalar_bool(info["success"]))
            if scalar_bool(terminated) or scalar_bool(truncated):
                break
        episodes.append(
            {
                "episode": episode_idx,
                "success": bool(successes[-1]) if successes else False,
                "length": len(rewards),
                "reward_sum": float(np.sum(rewards)),
                "final_phase": phases[-1] if phases else "",
                "phase_counts": {phase: phases.count(phase) for phase in sorted(set(phases))},
                "final_debug": final_debug,
                "terminated": scalar_bool(terminated),
                "truncated": scalar_bool(truncated),
            }
        )
    env.close()

    result = {
        "env_id": args.env_id,
        "control_mode": args.control_mode,
        "point_mode": args.point_mode,
        "episodes": episodes,
        "success_rate": float(np.mean([ep["success"] for ep in episodes])) if episodes else 0.0,
        "mean_length": float(np.mean([ep["length"] for ep in episodes])) if episodes else 0.0,
        "mean_reward_sum": float(np.mean([ep["reward_sum"] for ep in episodes])) if episodes else 0.0,
        "cube_seg_id": int(args.cube_seg_id),
        "gain": float(args.gain),
        "max_xyz_action": float(args.max_xyz_action),
        "note": "Geometry feasibility probe using pointcloud segmentation, tcp pose, and goal position.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("success_rate", "mean_length", "mean_reward_sum")}, indent=2))
    print(f"wrote {args.output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env_id", default="PickCube-v1")
    parser.add_argument("--control_mode", default="pd_ee_delta_pos")
    parser.add_argument("--point_mode", choices=["normal", "null", "cross_demo"], default="normal")
    parser.add_argument("--cross_hdf5", type=Path, default=None)
    parser.add_argument("--cross_samples", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--cube_seg_id", type=int, default=18)
    parser.add_argument("--gain", type=float, default=5.0)
    parser.add_argument("--max_xyz_action", type=float, default=0.25)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
