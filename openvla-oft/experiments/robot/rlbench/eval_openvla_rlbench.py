"""Evaluate OpenVLA/DepthVLA checkpoints in RLBench.

This runner uses RLBench directly instead of the BridgeVLA/RVT agent wrapper:

1. Reset each episode to a stored demo initial state.
2. Convert RLBench observations into the OpenVLA-OFT observation dict.
3. Predict one delta action from the model's action chunk.
4. Convert the delta action to an absolute RLBench EE-pose action.

It writes both CSV and JSON results so that
``compare_rgbd_rollout_results.py`` can apply the final RGB-D causal gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_depth_encoder,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)


DEFAULT_TASKS = (
    "slide_block_to_target",
    "turn_tap",
    "close_jar",
    "open_drawer",
    "reach_target",
    "pick_up_cup",
)


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


def jsonable_array(value: Any) -> list | float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


def extract_depth_debug(action_head) -> dict[str, np.ndarray]:
    if action_head is None:
        return {}
    module = action_head.module if hasattr(action_head, "module") else action_head
    debug: dict[str, np.ndarray] = {}
    for attr, key in (
        ("last_depth_point_xyz", "depth_point_xyz"),
        ("last_depth_waypoint_xyz_action", "depth_waypoint_xyz_action"),
    ):
        value = getattr(module, attr, None)
        if value is None:
            continue
        arr = value.detach().float().cpu().numpy().reshape(-1, 3)
        if arr.shape[0] > 0 and np.isfinite(arr[0]).all():
            debug[key] = arr[0].astype(np.float32)
    value = getattr(module, "last_depth_waypoint_chunk_xyz_action", None)
    if value is not None:
        arr = value.detach().float().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] > 0 and arr.shape[-1] == 3 and np.isfinite(arr[0]).all():
            debug["depth_waypoint_chunk_xyz_action"] = arr[0].astype(np.float32)
    return debug


class ClippedEndEffectorPoseViaPlanning(EndEffectorPoseViaPlanning):
    def action(self, scene, action: np.ndarray, ignore_collisions: bool = True):
        action = np.asarray(action, dtype=np.float32).copy()
        action[:3] = np.clip(
            action[:3],
            np.array([scene._workspace_minx, scene._workspace_miny, scene._workspace_minz], dtype=np.float32) + 1e-7,
            np.array([scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz], dtype=np.float32) - 1e-7,
        )
        return super().action(scene, action, ignore_collisions)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def quat_inverse(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    out = q.copy()
    out[:3] *= -1.0
    denom = float(np.dot(q, q))
    if denom <= 1e-8:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    return out / denom


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def euler_xyz_to_quat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in rpy]
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float32,
    )


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    return q / norm


def delta_to_absolute_action(
    gripper_pose: np.ndarray,
    delta: np.ndarray,
    max_delta_xyz: float,
    max_delta_rpy: float | None = None,
    gripper_override: float | None = None,
) -> np.ndarray:
    pose = np.asarray(gripper_pose, dtype=np.float32)[:7]
    delta = np.asarray(delta, dtype=np.float32).reshape(-1)
    delta_xyz = np.clip(delta[:3], -max_delta_xyz, max_delta_xyz)
    delta_rpy = delta[3:6]
    if max_delta_rpy is not None and max_delta_rpy >= 0:
        delta_rpy = np.clip(delta_rpy, -max_delta_rpy, max_delta_rpy)
    delta_quat = euler_xyz_to_quat(delta_rpy)
    next_xyz = pose[:3] + delta_xyz
    next_quat = normalize_quat(quat_multiply(delta_quat, pose[3:7]))
    gripper_value = delta[6] if gripper_override is None else gripper_override
    gripper_open = np.array([float(np.clip(gripper_value, 0.0, 1.0))], dtype=np.float32)
    return np.concatenate([next_xyz, next_quat, gripper_open]).astype(np.float32)


def gripper_override_for_step(args: argparse.Namespace, step: int) -> float | None:
    mode = str(args.gripper_override_mode or "none").lower()
    if mode == "none":
        return None
    if mode == "open":
        return 1.0
    if mode == "close":
        return 0.0
    if mode == "close_after_step":
        return 0.0 if step >= int(args.gripper_close_after_step) else 1.0
    if mode in ("close_near_depth_point", "latch_close_near_depth_point"):
        return None
    raise ValueError(f"Unknown gripper_override_mode: {args.gripper_override_mode!r}")


def gripper_override_for_geometry(
    args: argparse.Namespace,
    obs,
    depth_debug: dict[str, np.ndarray],
    latched_closed: bool,
) -> tuple[float | None, bool, float | None]:
    mode = str(args.gripper_override_mode or "none").lower()
    if mode not in ("close_near_depth_point", "latch_close_near_depth_point"):
        return None, latched_closed, None
    depth_point = depth_debug.get("depth_point_xyz")
    if depth_point is None:
        return 1.0, latched_closed, None
    ee_xyz = np.asarray(obs.gripper_pose[:3], dtype=np.float32)
    distance = float(np.linalg.norm(np.asarray(depth_point[:3], dtype=np.float32) - ee_xyz))
    should_close = distance <= float(args.gripper_close_distance)
    if mode == "latch_close_near_depth_point":
        latched_closed = latched_closed or should_close
        should_close = latched_closed
    return (0.0 if should_close else 1.0), latched_closed, distance


def parse_xyz(value: str) -> np.ndarray | None:
    value = str(value or "").strip()
    if not value:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated xyz values, got {value!r}")
    return np.asarray([float(part) for part in parts], dtype=np.float32)


def maybe_latch_depth_point(
    args: argparse.Namespace,
    depth_debug: dict[str, np.ndarray],
    latched_depth_point: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], np.ndarray | None, bool]:
    mode = str(args.depth_point_latch_mode or "none").lower()
    if mode == "none":
        return depth_debug, latched_depth_point, False
    if mode not in ("first", "demo_first_close"):
        raise ValueError(f"Unknown depth_point_latch_mode: {args.depth_point_latch_mode!r}")

    raw_depth_point = depth_debug.get("depth_point_xyz")
    if mode == "first" and latched_depth_point is None and raw_depth_point is not None:
        candidate = np.asarray(raw_depth_point[:3], dtype=np.float32)
        if np.isfinite(candidate).all():
            latched_depth_point = candidate.copy()

    if latched_depth_point is None:
        return depth_debug, latched_depth_point, False

    out = dict(depth_debug)
    if raw_depth_point is not None:
        out["raw_depth_point_xyz"] = np.asarray(raw_depth_point[:3], dtype=np.float32)
    out["depth_point_xyz"] = np.asarray(latched_depth_point[:3], dtype=np.float32)
    return out, latched_depth_point, True


def demo_first_close_xyz(demo: Any) -> np.ndarray | None:
    first_close = demo_first_close_index(demo)
    if first_close is not None:
        return np.asarray(demo[first_close].gripper_pose[:3], dtype=np.float32)
    if len(demo) > 0:
        return np.asarray(demo[-1].gripper_pose[:3], dtype=np.float32)
    return None


def demo_first_close_index(demo: Any) -> int | None:
    for idx, demo_obs in enumerate(demo):
        if float(getattr(demo_obs, "gripper_open", 1.0)) < 0.5:
            return idx
    return None


def demo_action(demo_obs: Any) -> np.ndarray:
    gripper_pose = np.asarray(demo_obs.gripper_pose[:7], dtype=np.float32)
    gripper_open = np.asarray([float(getattr(demo_obs, "gripper_open", 1.0))], dtype=np.float32)
    return np.concatenate([gripper_pose, gripper_open]).astype(np.float32)


def apply_latched_depth_point_action(
    args: argparse.Namespace,
    delta_action: np.ndarray,
    obs,
    latched_depth_point: np.ndarray | None,
) -> tuple[np.ndarray, bool, float | None]:
    max_step = float(args.latched_depth_point_action_step)
    if max_step <= 0 or latched_depth_point is None:
        return delta_action, False, None
    ee_xyz = np.asarray(obs.gripper_pose[:3], dtype=np.float32)
    vector = np.asarray(latched_depth_point[:3], dtype=np.float32) - ee_xyz
    distance = float(np.linalg.norm(vector))
    if not np.isfinite(distance) or distance <= 1e-8:
        return delta_action, False, distance
    step_xyz = vector if distance <= max_step else vector / distance * max_step
    out = delta_action.copy()
    out[:3] = step_xyz.astype(np.float32)
    if args.latched_depth_point_zero_rpy:
        out[3:6] = 0.0
    return out, True, distance


def camera_matrix(obs: Any, camera: str, suffix: str) -> np.ndarray:
    misc = getattr(obs, "misc", {}) or {}
    key = f"{camera}_{suffix}"
    if key not in misc:
        raise KeyError(f"RLBench observation is missing misc[{key!r}]")
    return np.asarray(misc[key], dtype=np.float32)


def depth_bundle_from_obs(obs: Any) -> dict[str, np.ndarray]:
    front_depth = np.asarray(obs.front_depth, dtype=np.float32)
    wrist_depth = np.asarray(obs.wrist_depth, dtype=np.float32)
    if front_depth.ndim == 3:
        front_depth = front_depth[..., 0]
    if wrist_depth.ndim == 3:
        wrist_depth = wrist_depth[..., 0]
    depth_values = np.stack([front_depth, wrist_depth]).astype(np.float32)
    return {
        "depth_values": depth_values,
        "depth_intrinsics": np.stack(
            [
                camera_matrix(obs, "front", "camera_intrinsics"),
                camera_matrix(obs, "wrist", "camera_intrinsics"),
            ]
        ).astype(np.float32),
        "depth_extrinsics": np.stack(
            [
                camera_matrix(obs, "front", "camera_extrinsics"),
                camera_matrix(obs, "wrist", "camera_extrinsics"),
            ]
        ).astype(np.float32),
        "depth_valid_mask": np.isfinite(depth_values) & (depth_values > 0),
    }


def corrupt_depth(depth_values: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    mode = str(mode or "none").lower()
    if mode in ("none", "normal"):
        return depth_values
    if mode in ("null", "zero"):
        return np.zeros_like(depth_values)
    if mode in ("shuffle", "shuffle_depth", "shuffle_pixels", "shuffle_tokens", "shuffle_geometry"):
        out = depth_values.copy()
        for view_idx in range(out.shape[0]):
            flat = out[view_idx].reshape(-1).copy()
            rng.shuffle(flat)
            out[view_idx] = flat.reshape(out[view_idx].shape)
        return out
    if mode in ("shuffle_samples", "cross_sample", "replace_from_other_episode", "replace_episode"):
        return depth_values
    raise ValueError(f"Unknown depth ablation mode: {mode}")


def rlbench_obs_to_policy_obs(
    obs: Any,
    depth_mode: str,
    rng: np.random.Generator,
    corrupt_source_obs: Any | None = None,
) -> dict[str, np.ndarray]:
    front_rgb = np.asarray(obs.front_rgb, dtype=np.uint8)
    wrist_rgb = np.asarray(obs.wrist_rgb, dtype=np.uint8)
    if str(depth_mode).lower() in ("shuffle_samples", "cross_sample", "replace_from_other_episode", "replace_episode"):
        if corrupt_source_obs is None:
            raise ValueError(f"depth_mode={depth_mode!r} requires corrupt_source_obs")
        depth_bundle = depth_bundle_from_obs(corrupt_source_obs)
    else:
        depth_bundle = depth_bundle_from_obs(obs)
        depth_bundle["depth_values"] = corrupt_depth(depth_bundle["depth_values"], depth_mode, rng)
        depth_bundle["depth_valid_mask"] = np.isfinite(depth_bundle["depth_values"]) & (depth_bundle["depth_values"] > 0)

    gripper_pose = np.asarray(obs.gripper_pose, dtype=np.float32)[:7]
    gripper_open = np.array([float(getattr(obs, "gripper_open", 1.0))], dtype=np.float32)
    proprio = np.concatenate([gripper_pose, gripper_open]).astype(np.float32)

    return {
        "full_image": front_rgb,
        "wrist_image": wrist_rgb,
        "state": proprio,
        **depth_bundle,
    }


def get_demo_obs_at(demo: Any, step: int) -> Any:
    index = min(max(0, int(step)), max(0, len(demo) - 1))
    return demo[index]


def load_corrupt_source_demo(args: argparse.Namespace, task_name: str, episode: int):
    if args.depth_mode not in ("shuffle_samples", "cross_sample", "replace_from_other_episode", "replace_episode"):
        return None
    source_episode = episode + int(args.depth_corrupt_episode_offset)
    try:
        return load_stored_demo(args.eval_datafolder, task_name, source_episode)
    except Exception:
        if not args.allow_same_episode_depth_corruption:
            raise
        return load_stored_demo(args.eval_datafolder, task_name, episode)


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


def make_model_cfg(args: argparse.Namespace) -> SimpleNamespace:
    depth_fusion_mode = args.depth_fusion_mode
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.checkpoint,
        base_model_checkpoint=args.base_model_checkpoint or args.checkpoint,
        processor_checkpoint=args.base_model_checkpoint or args.checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=args.center_crop,
        lora_rank=args.lora_rank,
        unnorm_key=args.unnorm_key,
        use_relative_actions=False,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_depth=args.use_depth,
        depth_fusion_mode=depth_fusion_mode,
        depth_encoder_type=args.depth_encoder_type,
        depth_num_points_per_view=args.depth_num_points_per_view,
        depth_hidden_dim=args.depth_hidden_dim,
        depth_grid_size=args.depth_grid_size,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        geometry_norm=args.geometry_norm,
        geometry_clip=args.geometry_clip,
        summary_repr="base_xyz",
        summary_pool="meanmax",
        depth_action_fusion_gate_init=1.0,
        depth_action_fusion_gate_override=args.depth_fusion_gate_override,
        depth_hidden_delta_clip=args.depth_hidden_delta_clip,
        depth_action_residual_clip=args.depth_action_residual_clip,
        depth_keypose_residual_weight=args.depth_keypose_residual_weight,
        depth_keypose_residual_clip=args.depth_keypose_residual_clip,
        depth_point_action_weight=args.depth_point_action_weight,
        depth_point_action_clip=args.depth_point_action_clip,
        depth_waypoint_action_weight=args.depth_waypoint_action_weight,
        depth_waypoint_action_clip=args.depth_waypoint_action_clip,
        depth_waypoint_action_scale=args.depth_waypoint_action_scale,
        depth_waypoint_action_chunk_len=args.depth_waypoint_action_chunk_len,
        depth_adapter_hidden_dim=256,
        aux_output_dim=args.aux_output_dim,
    )


def load_policy(args: argparse.Namespace):
    cfg = make_model_cfg(args)
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, 8)
    action_head = get_action_head(cfg, vla.llm_dim)
    depth_encoder = None
    if args.use_depth:
        depth_encoder = get_depth_encoder(cfg, vla.llm_dim)
        if hasattr(depth_encoder, "ablation_mode"):
            ablation = args.depth_mode
            if ablation in ("normal", "shuffle", "shuffle_samples", "cross_sample", "replace_from_other_episode"):
                ablation = "none"
            depth_encoder.ablation_mode = ablation
    return cfg, vla, processor, proprio_projector, action_head, depth_encoder


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cfg, vla, processor, proprio_projector, action_head, depth_encoder = load_policy(args)
    rng = np.random.default_rng(args.seed)
    trace_rows: list[dict[str, Any]] = []

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
    try:
        for task_name in task_names:
            task_cls = task_file_to_task_class(task_name)
            task_env = env.get_task(task_cls)
            rewards: list[float] = []
            lengths: list[int] = []
            errors: list[str] = []
            task_delta_xyz_norms: list[float] = []
            task_delta_rpy_norms: list[float] = []
            task_target_distances: list[float] = []
            task_gripper_commands: list[float] = []
            for episode in range(args.start_episode, args.start_episode + args.eval_episodes):
                demo = load_stored_demo(args.eval_datafolder, task_name, episode)
                corrupt_source_demo = load_corrupt_source_demo(args, task_name, episode) if args.use_depth else None
                task_env.set_variation(demo.variation_number)
                descriptions, obs = task_env.reset_to_demo(demo)
                instruction = descriptions[0] if descriptions else task_name.replace("_", " ")
                reward = 0.0
                error = ""
                step_count = 0
                delta_xyz_norms: list[float] = []
                delta_rpy_norms: list[float] = []
                target_distances: list[float] = []
                gripper_commands: list[float] = []
                cached_action_chunk: np.ndarray | None = None
                cached_depth_debug: dict[str, np.ndarray] = {}
                cached_action_index = 0
                chunk_exec_horizon = max(1, int(args.action_chunk_exec_horizon))
                gripper_latched_closed = False
                post_close_pull_delta_xyz = parse_xyz(args.post_close_pull_delta_xyz)
                post_close_latched_steps = 0
                post_close_pull_count = 0
                post_close_demo_tail_active = False
                post_close_demo_tail_count = 0
                demo_tail_start_index = None
                if str(args.post_close_demo_tail_mode or "none").lower() != "none":
                    first_close_index = demo_first_close_index(demo)
                    if first_close_index is not None:
                        demo_tail_start_index = max(
                            0,
                            int(first_close_index) - max(0, int(args.post_close_demo_tail_preclose_steps)),
                        )
                latched_depth_point: np.ndarray | None = None
                if str(args.depth_point_latch_mode or "none").lower() == "demo_first_close":
                    latched_depth_point = demo_first_close_xyz(demo)
                for step in range(args.episode_length):
                    step_count = step + 1
                    new_prediction = False
                    if cached_action_chunk is None or cached_action_index >= min(
                        chunk_exec_horizon, len(cached_action_chunk)
                    ):
                        corrupt_source_obs = (
                            get_demo_obs_at(corrupt_source_demo, step) if corrupt_source_demo is not None else None
                        )
                        policy_obs = rlbench_obs_to_policy_obs(
                            obs,
                            args.depth_mode if args.use_depth else "normal",
                            rng,
                            corrupt_source_obs=corrupt_source_obs,
                        )
                        cached_action_chunk = np.asarray(
                            get_vla_action(
                                cfg,
                                vla,
                                processor,
                                policy_obs,
                                instruction,
                                action_head=action_head,
                                proprio_projector=proprio_projector,
                                depth_encoder=depth_encoder,
                            ),
                            dtype=np.float32,
                        )
                        cached_depth_debug = extract_depth_debug(action_head)
                        cached_depth_debug, latched_depth_point, _ = maybe_latch_depth_point(
                            args,
                            cached_depth_debug,
                            latched_depth_point,
                        )
                        cached_action_index = 0
                        new_prediction = True
                    current_chunk_index = cached_action_index
                    delta_action = np.asarray(cached_action_chunk[current_chunk_index], dtype=np.float32)
                    cached_action_index += 1
                    depth_point_latched_active = (
                        str(args.depth_point_latch_mode or "none").lower() != "none"
                        and latched_depth_point is not None
                    )
                    delta_action, latched_depth_point_action_active, latched_depth_point_distance = (
                        apply_latched_depth_point_action(
                            args,
                            delta_action,
                            obs,
                            latched_depth_point,
                        )
                    )
                    step_override = gripper_override_for_step(args, step)
                    geometry_override, gripper_latched_closed, gripper_depth_distance = gripper_override_for_geometry(
                        args,
                        obs,
                        cached_depth_debug,
                        gripper_latched_closed,
                    )
                    gripper_override = geometry_override if geometry_override is not None else step_override
                    post_close_pull_active = False
                    if (
                        post_close_pull_delta_xyz is not None
                        and gripper_latched_closed
                        and post_close_pull_count < int(args.post_close_pull_steps)
                    ):
                        delay = max(0, int(args.post_close_pull_delay_steps))
                        if post_close_latched_steps >= delay:
                            delta_action = delta_action.copy()
                            delta_action[:3] = post_close_pull_delta_xyz
                            if args.post_close_pull_zero_rpy:
                                delta_action[3:6] = 0.0
                            post_close_pull_active = True
                            latched_depth_point_action_active = False
                            post_close_pull_count += 1
                    if gripper_latched_closed:
                        post_close_latched_steps += 1
                    rlbench_action = delta_to_absolute_action(
                        obs.gripper_pose,
                        delta_action,
                        args.max_delta_xyz,
                        args.max_delta_rpy,
                        gripper_override,
                    )
                    action_source = "policy"
                    post_close_demo_tail_index = None
                    post_close_demo_tail_action_active = False
                    if (
                        str(args.post_close_demo_tail_mode or "none").lower() == "first_close"
                        and gripper_latched_closed
                        and demo_tail_start_index is not None
                    ):
                        post_close_demo_tail_active = True
                        stride = max(1, int(args.post_close_demo_tail_stride))
                        post_close_demo_tail_index = min(
                            int(demo_tail_start_index) + post_close_demo_tail_count * stride,
                            len(demo) - 1,
                        )
                        rlbench_action = demo_action(demo[post_close_demo_tail_index])
                        post_close_demo_tail_count += 1
                        post_close_demo_tail_action_active = True
                        action_source = "demo_tail"
                    executed_delta_rpy = delta_action[3:6]
                    if args.max_delta_rpy is not None and args.max_delta_rpy >= 0:
                        executed_delta_rpy = np.clip(
                            executed_delta_rpy,
                            -args.max_delta_rpy,
                            args.max_delta_rpy,
                        )
                    executed_delta_xyz = rlbench_action[:3] - np.asarray(obs.gripper_pose[:3], dtype=np.float32)
                    delta_xyz_norms.append(float(np.linalg.norm(executed_delta_xyz)))
                    delta_rpy_norms.append(float(np.linalg.norm(executed_delta_rpy)))
                    target_distances.append(float(np.linalg.norm(rlbench_action[:3] - np.asarray(obs.gripper_pose[:3], dtype=np.float32))))
                    gripper_commands.append(float(rlbench_action[-1]))
                    if args.trace_output is not None and (
                        args.trace_max_steps <= 0 or len(trace_rows) < args.trace_max_steps
                    ):
                        depth_chunk = cached_depth_debug.get("depth_waypoint_chunk_xyz_action")
                        current_waypoint_chunk = None
                        if depth_chunk is not None and current_chunk_index < len(depth_chunk):
                            current_waypoint_chunk = depth_chunk[current_chunk_index]
                        trace_rows.append(
                            {
                                "task": task_name,
                                "episode": episode,
                                "step": step,
                                "depth_mode": args.depth_mode if args.use_depth else "rgb_only",
                                "new_prediction": new_prediction,
                                "chunk_index": current_chunk_index,
                                "action_chunk_exec_horizon": chunk_exec_horizon,
                                "action_source": action_source,
                                "ee_xyz_before": jsonable_array(np.asarray(obs.gripper_pose[:3], dtype=np.float32)),
                                "delta_action": jsonable_array(delta_action),
                                "executed_delta_xyz": jsonable_array(executed_delta_xyz),
                                "delta_xyz_norm": float(np.linalg.norm(executed_delta_xyz)),
                                "executed_delta_rpy": jsonable_array(executed_delta_rpy),
                                "target_xyz": jsonable_array(rlbench_action[:3]),
                                "target_distance": float(
                                    np.linalg.norm(rlbench_action[:3] - np.asarray(obs.gripper_pose[:3], dtype=np.float32))
                                ),
                                "gripper_command": float(rlbench_action[-1]),
                                "gripper_override_mode": args.gripper_override_mode,
                                "gripper_override_value": (
                                    None if gripper_override is None else float(gripper_override)
                                ),
                                "gripper_latched_closed": bool(gripper_latched_closed),
                                "gripper_depth_distance": gripper_depth_distance,
                                "post_close_pull_active": bool(post_close_pull_active),
                                "post_close_pull_step": int(post_close_pull_count)
                                if gripper_latched_closed
                                else None,
                                "post_close_latched_steps": int(post_close_latched_steps)
                                if gripper_latched_closed
                                else None,
                                "post_close_demo_tail_mode": args.post_close_demo_tail_mode,
                                "post_close_demo_tail_active": bool(post_close_demo_tail_active),
                                "post_close_demo_tail_action_active": bool(post_close_demo_tail_action_active),
                                "post_close_demo_tail_index": post_close_demo_tail_index,
                                "post_close_demo_tail_step": int(post_close_demo_tail_count)
                                if post_close_demo_tail_active
                                else None,
                                "depth_point_latch_mode": args.depth_point_latch_mode,
                                "depth_point_latched_active": bool(depth_point_latched_active),
                                "latched_depth_point_xyz": jsonable_array(latched_depth_point),
                                "raw_depth_point_xyz": jsonable_array(cached_depth_debug.get("raw_depth_point_xyz")),
                                "latched_depth_point_action_active": bool(latched_depth_point_action_active),
                                "latched_depth_point_distance": latched_depth_point_distance,
                                "depth_point_xyz": jsonable_array(cached_depth_debug.get("depth_point_xyz")),
                                "depth_waypoint_xyz_action": jsonable_array(
                                    cached_depth_debug.get("depth_waypoint_xyz_action")
                                ),
                                "depth_waypoint_chunk_xyz_action": jsonable_array(current_waypoint_chunk),
                            }
                        )
                    try:
                        obs, reward, terminal = task_env.step(rlbench_action)
                    except (IKError, ConfigurationPathError, InvalidActionError) as exc:
                        error = type(exc).__name__
                        reward = 0.0
                        terminal = True
                    if reward >= 1.0 or terminal:
                        break
                rewards.append(float(reward >= 1.0))
                lengths.append(step_count)
                errors.append(error)
                task_delta_xyz_norms.extend(delta_xyz_norms)
                task_delta_rpy_norms.extend(delta_rpy_norms)
                task_target_distances.extend(target_distances)
                task_gripper_commands.extend(gripper_commands)
                print(
                    f"[eval] task={task_name} episode={episode} "
                    f"success={int(reward >= 1.0)} length={step_count} error={error or '-'}"
                )
            rows.append(
                {
                    "task": task_name,
                    "success_rate": float(np.mean(rewards)) if rewards else 0.0,
                    "length": float(np.mean(lengths)) if lengths else 0.0,
                    "episodes": len(rewards),
                    "errors": {name: errors.count(name) for name in sorted(set(errors)) if name},
                    "delta_xyz_norm_mean": float(np.mean(task_delta_xyz_norms)) if task_delta_xyz_norms else 0.0,
                    "delta_xyz_norm_max": float(np.max(task_delta_xyz_norms)) if task_delta_xyz_norms else 0.0,
                    "delta_rpy_norm_mean": float(np.mean(task_delta_rpy_norms)) if task_delta_rpy_norms else 0.0,
                    "delta_rpy_norm_max": float(np.max(task_delta_rpy_norms)) if task_delta_rpy_norms else 0.0,
                    "target_distance_mean": float(np.mean(task_target_distances)) if task_target_distances else 0.0,
                    "target_distance_max": float(np.max(task_target_distances)) if task_target_distances else 0.0,
                    "gripper_command_mean": float(np.mean(task_gripper_commands)) if task_gripper_commands else 0.0,
                }
            )
    finally:
        env.shutdown()

    overall = float(np.mean([row["success_rate"] for row in rows])) if rows else 0.0
    return {
        "success_rate": overall,
        "depth_mode": args.depth_mode if args.use_depth else "rgb_only",
        "action_chunk_exec_horizon": max(1, int(args.action_chunk_exec_horizon)),
        "post_close_pull_delta_xyz": args.post_close_pull_delta_xyz,
        "post_close_pull_steps": int(args.post_close_pull_steps),
        "post_close_pull_delay_steps": int(args.post_close_pull_delay_steps),
        "post_close_demo_tail_mode": args.post_close_demo_tail_mode,
        "post_close_demo_tail_preclose_steps": int(args.post_close_demo_tail_preclose_steps),
        "post_close_demo_tail_stride": int(args.post_close_demo_tail_stride),
        "depth_point_latch_mode": args.depth_point_latch_mode,
        "latched_depth_point_action_step": float(args.latched_depth_point_action_step),
        "tasks": {row["task"]: row["success_rate"] for row in rows},
        "task_results": {row["task"]: row for row in rows},
        "checkpoint": args.checkpoint,
        "eval_datafolder": args.eval_datafolder,
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
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned checkpoint/run directory.")
    parser.add_argument("--base_model_checkpoint", default="", help="Base OpenVLA checkpoint for processor/model code.")
    parser.add_argument("--eval_datafolder", required=True, help="RLBench raw demo dataset root.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--unnorm_key", required=True)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=25)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--center_crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument(
        "--depth_mode",
        choices=("normal", "null", "shuffle", "shuffle_samples", "cross_sample", "replace_from_other_episode"),
        default="normal",
    )
    parser.add_argument(
        "--depth_corrupt_episode_offset",
        type=int,
        default=1,
        help="Episode offset used by cross-sample depth corruption modes.",
    )
    parser.add_argument(
        "--allow_same_episode_depth_corruption",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fall back to the current episode if the requested cross-sample source episode is missing.",
    )
    parser.add_argument("--depth_fusion_mode", default="object_query")
    parser.add_argument("--depth_encoder_type", choices=("grid", "dense_point"), default="dense_point")
    parser.add_argument("--depth_num_points_per_view", type=int, default=1024)
    parser.add_argument("--depth_hidden_dim", type=int, default=256)
    parser.add_argument("--depth_grid_size", type=int, default=4)
    parser.add_argument("--depth_min_m", type=float, default=0.01)
    parser.add_argument("--depth_max_m", type=float, default=5.0)
    parser.add_argument("--geometry_norm", default="none")
    parser.add_argument("--geometry_clip", type=float, default=5.0)
    parser.add_argument("--depth_fusion_gate_override", type=float, default=None)
    parser.add_argument("--depth_hidden_delta_clip", type=float, default=0.0)
    parser.add_argument("--depth_action_residual_clip", type=float, default=0.0)
    parser.add_argument("--depth_keypose_residual_weight", type=float, default=0.0)
    parser.add_argument("--depth_keypose_residual_clip", type=float, default=0.0)
    parser.add_argument("--depth_point_action_weight", type=float, default=0.0)
    parser.add_argument("--depth_point_action_clip", type=float, default=0.0)
    parser.add_argument("--depth_waypoint_action_weight", type=float, default=0.0)
    parser.add_argument("--depth_waypoint_action_clip", type=float, default=0.0)
    parser.add_argument("--depth_waypoint_action_scale", type=float, default=1.0)
    parser.add_argument("--depth_waypoint_action_chunk_len", type=int, default=1)
    parser.add_argument("--aux_output_dim", type=int, default=8)
    parser.add_argument("--max_delta_xyz", type=float, default=0.08)
    parser.add_argument(
        "--max_delta_rpy",
        type=float,
        default=None,
        help="Optional per-axis rpy delta clamp in radians; omit to preserve raw model orientation deltas.",
    )
    parser.add_argument(
        "--gripper_override_mode",
        choices=(
            "none",
            "open",
            "close",
            "close_after_step",
            "close_near_depth_point",
            "latch_close_near_depth_point",
        ),
        default="none",
        help="Eval-only gripper diagnostic override; not used for final learned-policy claims.",
    )
    parser.add_argument("--gripper_close_after_step", type=int, default=75)
    parser.add_argument(
        "--gripper_close_distance",
        type=float,
        default=0.03,
        help="EE-to-selected-depth-point threshold for depth-near gripper diagnostic modes.",
    )
    parser.add_argument(
        "--action_chunk_exec_horizon",
        type=int,
        default=1,
        help="Number of predicted action chunk steps to execute before recomputing the policy.",
    )
    parser.add_argument(
        "--depth_point_latch_mode",
        choices=("none", "first", "demo_first_close"),
        default="none",
        help=(
            "Eval-only diagnostic: latch the selected depth point for downstream gripper/action diagnostics. "
            "This is not used for learned-policy claims."
        ),
    )
    parser.add_argument(
        "--latched_depth_point_action_step",
        type=float,
        default=0.0,
        help=(
            "Eval-only diagnostic max xyz step in meters toward the latched depth point; <=0 leaves "
            "the learned xyz action unchanged."
        ),
    )
    parser.add_argument(
        "--latched_depth_point_zero_rpy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero orientation deltas while using the eval-only latched-depth-point xyz controller.",
    )
    parser.add_argument(
        "--post_close_pull_delta_xyz",
        default="",
        help=(
            "Eval-only diagnostic xyz delta to execute for a few steps after the depth-near "
            "gripper latch closes; leave empty to disable."
        ),
    )
    parser.add_argument(
        "--post_close_pull_steps",
        type=int,
        default=0,
        help="Number of post-close diagnostic pull steps to execute when post_close_pull_delta_xyz is set.",
    )
    parser.add_argument(
        "--post_close_pull_delay_steps",
        type=int,
        default=0,
        help="Optional delay after the depth-near gripper latch before post-close pull starts.",
    )
    parser.add_argument(
        "--post_close_pull_zero_rpy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero orientation deltas during the eval-only post-close pull diagnostic.",
    )
    parser.add_argument(
        "--post_close_demo_tail_mode",
        choices=("none", "first_close"),
        default="none",
        help=(
            "Eval-only oracle diagnostic: after the depth-near gripper latch triggers, execute stored demo "
            "absolute EE poses. This is not a learned-policy result."
        ),
    )
    parser.add_argument(
        "--post_close_demo_tail_preclose_steps",
        type=int,
        default=0,
        help="Number of demo steps before the first close pose to start the eval-only demo tail from.",
    )
    parser.add_argument(
        "--post_close_demo_tail_stride",
        type=int,
        default=1,
        help="Stride through stored demo poses for the eval-only demo tail.",
    )
    parser.add_argument("--trace_output", type=Path, default=None, help="Optional JSONL per-step trace output.")
    parser.add_argument(
        "--trace_max_steps",
        type=int,
        default=0,
        help="Maximum trace rows to write; <=0 means all evaluated steps.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    results = evaluate(args)
    trace_rows = results.pop("trace_rows", [])
    write_outputs(results, args.output)
    write_trace(trace_rows, args.trace_output)


if __name__ == "__main__":
    main()
