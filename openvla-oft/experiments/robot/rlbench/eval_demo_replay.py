"""Replay stored RLBench demonstration EE poses with the current eval action mode.

This is an oracle diagnostic, not a learned-policy result. It answers whether
the current RLBench reset/action stack can solve a task when given the stored
demo end-effector pose and gripper-open sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import task as rlbench_task
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment


DEFAULT_TASKS = ("open_drawer",)


class ClippedEndEffectorPoseViaPlanning(EndEffectorPoseViaPlanning):
    def action(self, scene, action: np.ndarray, ignore_collisions: bool = True):
        action = np.asarray(action, dtype=np.float32).copy()
        action[:3] = np.clip(
            action[:3],
            np.array([scene._workspace_minx, scene._workspace_miny, scene._workspace_minz], dtype=np.float32) + 1e-7,
            np.array([scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz], dtype=np.float32) - 1e-7,
        )
        return super().action(scene, action, ignore_collisions)


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
    raise FileNotFoundError(f"Could not find all_variations episodes for task={task_name!r}; tried {candidates}")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def jsonable_array(value: Any) -> list | float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


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


def demo_action(demo_obs) -> np.ndarray:
    gripper_pose = np.asarray(demo_obs.gripper_pose[:7], dtype=np.float32)
    gripper_open = np.asarray([float(getattr(demo_obs, "gripper_open", 1.0))], dtype=np.float32)
    return np.concatenate([gripper_pose, gripper_open]).astype(np.float32)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
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
        action_mode=MoveArmThenGripper(ClippedEndEffectorPoseViaPlanning(), Discrete()),
        obs_config=make_obs_config(args.image_size),
        dataset_root=args.eval_datafolder,
        headless=args.headless,
    )
    env.launch()

    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    try:
        for task_name in task_names:
            task_cls = task_file_to_task_class(task_name)
            task_env = env.get_task(task_cls)
            rewards: list[float] = []
            lengths: list[int] = []
            errors: list[str] = []
            for episode in range(args.start_episode, args.start_episode + args.eval_episodes):
                demo = load_stored_demo(args.eval_datafolder, task_name, episode)
                task_env.set_variation(demo.variation_number)
                _, obs = task_env.reset_to_demo(demo)
                reward = 0.0
                error = ""
                step_count = 0
                max_steps = args.episode_length if args.episode_length > 0 else max(1, len(demo) - 1)
                for step in range(max_steps):
                    step_count = step + 1
                    demo_index = min(int(args.start_demo_index) + step * max(1, int(args.demo_stride)), len(demo) - 1)
                    action = demo_action(demo[demo_index])
                    if args.force_gripper_open is not None:
                        action[-1] = float(args.force_gripper_open)
                    if args.trace_output is not None and (
                        args.trace_max_steps <= 0 or len(trace_rows) < args.trace_max_steps
                    ):
                        trace_rows.append(
                            {
                                "task": task_name,
                                "episode": episode,
                                "step": step,
                                "demo_index": demo_index,
                                "ee_xyz_before": jsonable_array(np.asarray(obs.gripper_pose[:3], dtype=np.float32)),
                                "target_xyz": jsonable_array(action[:3]),
                                "target_quat": jsonable_array(action[3:7]),
                                "target_gripper": float(action[-1]),
                                "target_distance": float(
                                    np.linalg.norm(action[:3] - np.asarray(obs.gripper_pose[:3], dtype=np.float32))
                                ),
                            }
                        )
                    try:
                        obs, reward, terminal = task_env.step(action)
                    except (IKError, ConfigurationPathError, InvalidActionError) as exc:
                        error = type(exc).__name__
                        reward = 0.0
                        terminal = True
                    if reward >= 1.0 or terminal:
                        break
                rewards.append(float(reward >= 1.0))
                lengths.append(step_count)
                errors.append(error)
                print(
                    f"[demo-replay] task={task_name} episode={episode} "
                    f"success={int(reward >= 1.0)} length={step_count} error={error or '-'}"
                )
            rows.append(
                {
                    "task": task_name,
                    "success_rate": float(np.mean(rewards)) if rewards else 0.0,
                    "length": float(np.mean(lengths)) if lengths else 0.0,
                    "episodes": len(rewards),
                    "errors": {name: errors.count(name) for name in sorted(set(errors)) if name},
                }
            )
    finally:
        env.shutdown()

    overall = float(np.mean([row["success_rate"] for row in rows])) if rows else 0.0
    return {
        "success_rate": overall,
        "tasks": {row["task"]: row["success_rate"] for row in rows},
        "task_results": {row["task"]: row for row in rows},
        "eval_datafolder": args.eval_datafolder,
        "start_demo_index": int(args.start_demo_index),
        "demo_stride": int(args.demo_stride),
        "trace_rows": trace_rows if args.trace_output is not None else [],
    }


def write_outputs(results: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(results, f, indent=2)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "success rate", "length", "episodes"])
        writer.writeheader()
        for row in results["task_results"].values():
            writer.writerow(
                {
                    "task": row["task"],
                    "success rate": row["success_rate"],
                    "length": row["length"],
                    "episodes": row["episodes"],
                }
            )
    print(f"[done] wrote {output}")
    print(f"[done] wrote {csv_path}")


def write_trace(trace_rows: list[dict[str, Any]], trace_output: Path | None) -> None:
    if trace_output is None:
        return
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    with trace_output.open("w") as f:
        for row in trace_rows:
            f.write(json.dumps(row) + "\n")
    print(f"[done] wrote {trace_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_datafolder", required=True, help="RLBench raw demo dataset root.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start_demo_index", type=int, default=1)
    parser.add_argument("--demo_stride", type=int, default=1)
    parser.add_argument("--force_gripper_open", type=float, default=None)
    parser.add_argument("--trace_output", type=Path, default=None)
    parser.add_argument("--trace_max_steps", type=int, default=0)
    args = parser.parse_args()

    results = evaluate(args)
    trace_rows = results.pop("trace_rows", [])
    write_outputs(results, args.output)
    write_trace(trace_rows, args.trace_output)


if __name__ == "__main__":
    main()
