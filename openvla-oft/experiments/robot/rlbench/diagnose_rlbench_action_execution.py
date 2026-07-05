"""Diagnose whether converted RLBench actions are executable.

This script replays stored demonstration actions through the same RLBench action
mode used by ``eval_openvla_rlbench.py``. It separates policy-learning failure
from action-mode, quaternion, scale, or reset-alignment bugs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

if os.environ.get("COPPELIASIM_ROOT"):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]
    os.environ["QT_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]

from pyrep.const import RenderMode
from pyrep.errors import ConfigurationPathError, IKError
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import task as rlbench_task
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robot.rlbench.convert_rlbench_to_hdf5 import abs_keypose_action, delta_action
from experiments.robot.rlbench.eval_openvla_rlbench import ClippedEndEffectorPoseViaPlanning, delta_to_absolute_action


DEFAULT_TASKS = (
    "slide_block_to_target",
    "turn_tap",
    "close_jar",
    "open_drawer",
    "reach_target",
    "pick_up_cup",
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_stored_demo(data_root: str | Path, task_name: str, episode: int):
    from peract_colab.rlbench.utils import get_stored_demo

    data_root = Path(data_root)
    candidates = (
        data_root / task_name / "all_variations" / "episodes",
        data_root / "train" / task_name / "all_variations" / "episodes",
        data_root / "val" / task_name / "all_variations" / "episodes",
    )
    for episodes_dir in candidates:
        if episodes_dir.exists():
            return get_stored_demo(data_path=str(episodes_dir), index=episode)
    raise FileNotFoundError(f"Could not find episodes for task={task_name!r}; tried {candidates}")


def make_obs_config(image_size: int) -> ObservationConfig:
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    for camera in (obs_config.front_camera, obs_config.wrist_camera):
        camera.set_all(True)
        camera.image_size = [image_size, image_size]
        camera.depth_in_meters = True
        camera.masks_as_one_channel = False
        camera.render_mode = RenderMode.OPENGL3
    obs_config.gripper_pose = True
    obs_config.gripper_open = True
    obs_config.gripper_matrix = True
    obs_config.joint_positions = True
    obs_config.wrist_camera_matrix = True
    return obs_config


def make_action_mode(name: str):
    if name == "planning":
        arm_mode = ClippedEndEffectorPoseViaPlanning()
    elif name == "ik":
        arm_mode = EndEffectorPoseViaIK()
    else:
        raise ValueError("--arm_action_mode must be planning|ik")
    return MoveArmThenGripper(arm_mode, Discrete())


def demo_delta_stats(demo) -> dict[str, float]:
    deltas = np.stack([delta_action(demo[t], demo[t + 1]) for t in range(len(demo) - 1)]).astype(np.float32)
    xyz_norm = np.linalg.norm(deltas[:, :3], axis=-1)
    rpy_norm = np.linalg.norm(deltas[:, 3:6], axis=-1)
    return {
        "num_transitions": int(deltas.shape[0]),
        "xyz_mean": float(xyz_norm.mean()),
        "xyz_p95": float(np.percentile(xyz_norm, 95)),
        "xyz_max": float(xyz_norm.max()),
        "rpy_mean": float(rpy_norm.mean()),
        "rpy_p95": float(np.percentile(rpy_norm, 95)),
        "rpy_max": float(rpy_norm.max()),
    }


def replay_episode(task_env, demo, mode: str, max_steps: int, max_delta_xyz: float) -> dict[str, Any]:
    task_env.set_variation(demo.variation_number)
    _, obs = task_env.reset_to_demo(demo)
    errors: list[str] = []
    rewards: list[float] = []
    distances: list[float] = []
    steps = min(max_steps, len(demo) - 1)

    for t in range(steps):
        next_demo_obs = demo[t + 1]
        if mode == "next_abs":
            action = abs_keypose_action(next_demo_obs)
        elif mode == "delta_to_abs_demo":
            demo_delta = delta_action(demo[t], next_demo_obs)
            action = delta_to_absolute_action(obs.gripper_pose, demo_delta, max_delta_xyz=max_delta_xyz)
            action[-1] = float(getattr(next_demo_obs, "gripper_open", action[-1]))
        else:
            raise ValueError("--mode must be next_abs|delta_to_abs_demo")

        target_xyz = np.asarray(action[:3], dtype=np.float32)
        before_xyz = np.asarray(obs.gripper_pose[:3], dtype=np.float32)
        distances.append(float(np.linalg.norm(target_xyz - before_xyz)))
        try:
            obs, reward, terminal = task_env.step(action.astype(np.float32))
        except (IKError, ConfigurationPathError, InvalidActionError) as exc:
            errors.append(type(exc).__name__)
            return {
                "success": 0.0,
                "terminal": True,
                "steps": t + 1,
                "error": type(exc).__name__,
                "mean_target_distance": float(np.mean(distances)) if distances else 0.0,
                "max_target_distance": float(np.max(distances)) if distances else 0.0,
                "rewards": rewards,
            }
        rewards.append(float(reward))
        if reward >= 1.0 or terminal:
            return {
                "success": float(reward >= 1.0),
                "terminal": bool(terminal),
                "steps": t + 1,
                "error": "",
                "mean_target_distance": float(np.mean(distances)) if distances else 0.0,
                "max_target_distance": float(np.max(distances)) if distances else 0.0,
                "rewards": rewards,
            }

    return {
        "success": float(rewards[-1] >= 1.0) if rewards else 0.0,
        "terminal": False,
        "steps": steps,
        "error": "",
        "mean_target_distance": float(np.mean(distances)) if distances else 0.0,
        "max_target_distance": float(np.max(distances)) if distances else 0.0,
        "rewards": rewards,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    task_names = parse_csv(args.tasks) or list(DEFAULT_TASKS)
    task_files = {
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    }
    for task_name in task_names:
        if task_name not in task_files:
            raise ValueError(f"Task {task_name!r} not recognised by RLBench")

    env = Environment(
        action_mode=make_action_mode(args.arm_action_mode),
        obs_config=make_obs_config(args.image_size),
        dataset_root=args.eval_datafolder,
        headless=args.headless,
    )
    env.launch()

    rows: list[dict[str, Any]] = []
    try:
        for task_name in task_names:
            task_env = env.get_task(task_file_to_task_class(task_name))
            for episode in range(args.start_episode, args.start_episode + args.eval_episodes):
                demo = load_stored_demo(args.eval_datafolder, task_name, episode)
                stats = demo_delta_stats(demo)
                result = replay_episode(task_env, demo, args.mode, args.max_steps, args.max_delta_xyz)
                row = {
                    "task": task_name,
                    "episode": episode,
                    "variation": int(demo.variation_number),
                    "mode": args.mode,
                    "arm_action_mode": args.arm_action_mode,
                    **stats,
                    **result,
                }
                rows.append(row)
                print(
                    f"[replay] task={task_name} ep={episode} mode={args.mode} "
                    f"success={int(row['success'])} steps={row['steps']} error={row['error'] or '-'} "
                    f"xyz_p95={row['xyz_p95']:.4f} max_target_dist={row['max_target_distance']:.4f}"
                )
    finally:
        env.shutdown()

    return {
        "eval_datafolder": args.eval_datafolder,
        "mode": args.mode,
        "arm_action_mode": args.arm_action_mode,
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "error_counts": {name: [row["error"] for row in rows].count(name) for name in sorted({row["error"] for row in rows}) if name},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_datafolder", required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=25)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mode", choices=("next_abs", "delta_to_abs_demo"), default="next_abs")
    parser.add_argument("--arm_action_mode", choices=("planning", "ik"), default="planning")
    parser.add_argument("--max_delta_xyz", type=float, default=0.08)
    parser.add_argument("--output", type=Path, default=Path("experiments/logs/rlbench_action_replay_diag.json"))
    args = parser.parse_args()

    results = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
