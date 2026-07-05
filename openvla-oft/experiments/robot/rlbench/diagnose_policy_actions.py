"""Offline policy-vs-demo action diagnostic for converted RLBench HDF5 data.

This does not launch RLBench/CoppeliaSim. It loads a trained OpenVLA/DepthVLA
checkpoint, predicts the first action for sampled HDF5 observations, and
compares it with the stored ``rlbench_delta_action`` target.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch

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


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_hdf5_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def iter_demo_keys(file_obj: h5py.File) -> list[str]:
    def key_order(key: str) -> int:
        try:
            return int(key.split("_")[-1])
        except ValueError:
            return 10**9

    return sorted(file_obj["data"].keys(), key=key_order)


def list_hdf5_files(data_dir: Path, tasks: set[str] | None) -> list[Path]:
    files = sorted(list(data_dir.glob("*.hdf5")) + list(data_dir.glob("*.h5")))
    if tasks:
        files = [path for path in files if any(task in path.stem for task in tasks)]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir} for tasks={sorted(tasks) if tasks else 'all'}")
    return files


def corrupt_depth(depth_values: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    mode = str(mode or "normal").lower()
    if mode in ("normal", "none"):
        return depth_values
    if mode in ("null", "zero"):
        return np.zeros_like(depth_values)
    if mode in ("shuffle", "shuffle_depth", "shuffle_pixels", "shuffle_geometry"):
        out = depth_values.copy()
        for view_idx in range(out.shape[0]):
            flat = out[view_idx].reshape(-1).copy()
            rng.shuffle(flat)
            out[view_idx] = flat.reshape(out[view_idx].shape)
        return out
    if mode in ("shuffle_samples", "cross_sample", "replace_from_other_episode", "replace_episode"):
        return depth_values
    raise ValueError(f"Unknown depth mode: {mode}")


def needs_depth_bank(depth_mode: str) -> bool:
    return str(depth_mode or "").lower() in (
        "shuffle_samples",
        "cross_sample",
        "replace_from_other_episode",
        "replace_episode",
    )


def visible_workspace_points(obs: h5py.Group, t: int) -> np.ndarray:
    point_sets = []
    for key in ("agentview_point_cloud", "eye_in_hand_point_cloud"):
        if key not in obs:
            continue
        cloud = np.asarray(obs[key][t], dtype=np.float32).reshape(-1, 3)
        valid = np.isfinite(cloud).all(axis=1)
        workspace = (
            valid
            & (cloud[:, 2] > 0.65)
            & (cloud[:, 2] < 1.35)
            & (cloud[:, 0] > -0.4)
            & (cloud[:, 0] < 1.1)
            & (cloud[:, 1] > -0.9)
            & (cloud[:, 1] < 0.9)
        )
        cloud = cloud[workspace]
        if cloud.size:
            point_sets.append(cloud)
    if not point_sets:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(point_sets, axis=0).astype(np.float32)


def nearest_visible_point(obs: h5py.Group, t: int, target_xyz: np.ndarray) -> np.ndarray:
    target = np.asarray(target_xyz, dtype=np.float32).reshape(1, 3)
    points = visible_workspace_points(obs, t)
    if points.shape[0] == 0:
        return target.reshape(3).astype(np.float32)
    idx = int(np.linalg.norm(points - target, axis=1).argmin())
    return points[idx].astype(np.float32)


def compute_aux_xyz_label(demo: h5py.Group, t: int, aux_target: str, future_horizon: int) -> np.ndarray | None:
    aux_target = str(aux_target or "").strip()
    if aux_target in ("", "none"):
        return None
    obs = demo["obs"]
    if "rlbench_abs_gripper_pose" not in obs or "proprio" not in obs:
        return None
    abs_xyz = np.asarray(obs["rlbench_abs_gripper_pose"][:, :3], dtype=np.float32)
    gripper_open = np.asarray(obs["proprio"][:, -1], dtype=np.float32)
    close_indices = np.flatnonzero(gripper_open < 0.5)
    close_index = int(close_indices[0]) if close_indices.size else abs_xyz.shape[0] - 1
    if aux_target in ("pre_first_close_pose_xyz", "visible_pre_first_close_point_xyz"):
        close_index = max(0, close_index - max(1, int(future_horizon)))
    if aux_target in (
        "first_close_pose_xyz",
        "pre_first_close_pose_xyz",
        "visible_first_close_point_xyz",
        "visible_pre_first_close_point_xyz",
    ):
        target = abs_xyz[close_index].astype(np.float32)
        if aux_target in ("visible_first_close_point_xyz", "visible_pre_first_close_point_xyz"):
            return nearest_visible_point(obs, t, target)
        return target
    return None


def make_model_cfg(args: argparse.Namespace) -> SimpleNamespace:
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
        depth_fusion_mode=args.depth_fusion_mode,
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
            depth_encoder.ablation_mode = (
                "none"
                if args.depth_mode in ("normal", "shuffle", "shuffle_samples", "cross_sample", "replace_from_other_episode")
                else args.depth_mode
            )
    return cfg, vla, processor, proprio_projector, action_head, depth_encoder


def sample_indices(length: int, stride: int) -> range:
    return range(0, length, max(1, stride))


def collect_depth_bank(
    data_dir: Path,
    tasks: set[str] | None,
    max_bank_samples: int,
    stride: int,
) -> list[dict[str, np.ndarray]]:
    bank: list[dict[str, np.ndarray]] = []
    for file_path in list_hdf5_files(data_dir, tasks):
        with h5py.File(file_path, "r") as f:
            for demo_key in iter_demo_keys(f):
                obs = f["data"][demo_key]["obs"]
                length = int(obs["agentview_depth_m"].shape[0])
                for t in sample_indices(length, stride):
                    depth_values = np.stack([obs["agentview_depth_m"][t], obs["eye_in_hand_depth_m"][t]], axis=0).astype(
                        np.float32
                    )
                    bank.append(
                        {
                            "depth_values": depth_values,
                            "depth_intrinsics": np.stack([obs["agentview_K"][t], obs["eye_in_hand_K"][t]], axis=0).astype(
                                np.float32
                            ),
                            "depth_extrinsics": np.stack(
                                [obs["agentview_T_camera_to_base"][t], obs["eye_in_hand_T_camera_to_base"][t]], axis=0
                            ).astype(np.float32),
                            "depth_valid_mask": np.isfinite(depth_values) & (depth_values > 0),
                        }
                    )
                    if max_bank_samples and len(bank) >= max_bank_samples:
                        return bank
    if not bank:
        raise RuntimeError("Depth corruption bank is empty")
    return bank


def build_obs(
    demo: h5py.Group,
    t: int,
    use_depth: bool,
    depth_mode: str,
    rng: np.random.Generator,
    depth_bank: list[dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    obs = demo["obs"]
    proprio = np.asarray(obs["proprio"][t], dtype=np.float32)
    out: dict[str, Any] = {
        "full_image": np.asarray(obs["agentview_rgb"][t], dtype=np.uint8),
        "wrist_image": np.asarray(obs["eye_in_hand_rgb"][t], dtype=np.uint8),
        "state": proprio,
    }
    if use_depth:
        if needs_depth_bank(depth_mode):
            if not depth_bank:
                raise ValueError(f"depth_mode={depth_mode!r} requires a non-empty depth_bank")
            source = depth_bank[int(rng.integers(len(depth_bank)))]
            out.update({key: np.asarray(value) for key, value in source.items()})
        else:
            depth_values = np.stack([obs["agentview_depth_m"][t], obs["eye_in_hand_depth_m"][t]], axis=0).astype(np.float32)
            depth_values = corrupt_depth(depth_values, depth_mode, rng)
            out.update(
                {
                    "depth_values": depth_values,
                    "depth_intrinsics": np.stack([obs["agentview_K"][t], obs["eye_in_hand_K"][t]], axis=0).astype(
                        np.float32
                    ),
                    "depth_extrinsics": np.stack(
                        [obs["agentview_T_camera_to_base"][t], obs["eye_in_hand_T_camera_to_base"][t]], axis=0
                    ).astype(np.float32),
                    "depth_valid_mask": np.isfinite(depth_values) & (depth_values > 0),
                }
            )
    return out


def action_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)[:7]
    target = np.asarray(target, dtype=np.float32).reshape(-1)[:7]
    diff = pred - target
    xyz_pred = pred[:3]
    xyz_target = target[:3]
    denom = float(np.linalg.norm(xyz_pred) * np.linalg.norm(xyz_target))
    xyz_cos = float(np.dot(xyz_pred, xyz_target) / denom) if denom > 1e-8 else 0.0
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "xyz_mae": float(np.mean(np.abs(diff[:3]))),
        "xyz_rmse": float(np.sqrt(np.mean(diff[:3] ** 2))),
        "rpy_mae": float(np.mean(np.abs(diff[3:6]))),
        "rpy_rmse": float(np.sqrt(np.mean(diff[3:6] ** 2))),
        "gripper_abs_error": float(abs(diff[6])),
        "pred_xyz_norm": float(np.linalg.norm(xyz_pred)),
        "target_xyz_norm": float(np.linalg.norm(xyz_target)),
        "xyz_direction_cosine": xyz_cos,
    }


def paired_action_delta_metrics(pred: np.ndarray, other_pred: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)[:7]
    other_pred = np.asarray(other_pred, dtype=np.float32).reshape(-1)[:7]
    diff = pred - other_pred
    return {
        "paired_pred_l1": float(np.mean(np.abs(diff))),
        "paired_pred_rmse": float(np.sqrt(np.mean(diff**2))),
        "paired_pred_xyz_l2": float(np.linalg.norm(diff[:3])),
        "paired_pred_rpy_l2": float(np.linalg.norm(diff[3:6])),
        "paired_pred_gripper_abs": float(abs(diff[6])),
    }


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


def depth_debug_scalar_metrics(debug: dict[str, np.ndarray]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in debug.items():
        metrics[f"{key}_norm"] = float(np.linalg.norm(value))
        if value.shape == (3,):
            for idx, axis in enumerate(("x", "y", "z")):
                metrics[f"{key}_{axis}"] = float(value[idx])
        elif value.ndim == 2 and value.shape[-1] == 3:
            step_norms = np.linalg.norm(value, axis=-1)
            metrics[f"{key}_step_norm_mean"] = float(step_norms.mean())
            metrics[f"{key}_step_norm_max"] = float(step_norms.max())
            for step_idx, step_norm in enumerate(step_norms[:8]):
                metrics[f"{key}_step{step_idx}_norm"] = float(step_norm)
    return metrics


def paired_depth_debug_metrics(debug: dict[str, np.ndarray], other_debug: dict[str, np.ndarray]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in debug.items():
        other = other_debug.get(key)
        if other is None:
            continue
        metrics[f"paired_{key}_l2"] = float(np.linalg.norm(value - other))
        metrics[f"paired_{key}_l1"] = float(np.mean(np.abs(value - other)))
    return metrics


def aux_label_depth_point_metrics(
    debug: dict[str, np.ndarray],
    aux_label: np.ndarray | None,
    prefix: str = "",
) -> dict[str, float]:
    if aux_label is None or "depth_point_xyz" not in debug:
        return {}
    target = np.asarray(aux_label, dtype=np.float32).reshape(3)
    point = np.asarray(debug["depth_point_xyz"], dtype=np.float32).reshape(3)
    diff = point - target
    metrics = {
        f"{prefix}depth_point_to_aux_label_l2": float(np.linalg.norm(diff)),
        f"{prefix}depth_point_to_aux_label_l1": float(np.mean(np.abs(diff))),
        f"{prefix}aux_label_xyz_norm": float(np.linalg.norm(target)),
    }
    for idx, axis in enumerate(("x", "y", "z")):
        metrics[f"{prefix}aux_label_xyz_{axis}"] = float(target[idx])
    return metrics


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No diagnostic rows collected")
    metric_keys = [key for key, value in rows[0]["metrics"].items() if isinstance(value, (int, float))]
    summary = {
        key: float(np.mean([row["metrics"][key] for row in rows]))
        for key in metric_keys
    }
    summary.update(
        {
            f"{key}_p95": float(np.percentile([row["metrics"][key] for row in rows], 95))
            for key in ("xyz_rmse", "pred_xyz_norm", "target_xyz_norm")
        }
    )
    by_task: dict[str, dict[str, float]] = {}
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        by_task[task] = {
            "samples": len(task_rows),
            "xyz_rmse": float(np.mean([row["metrics"]["xyz_rmse"] for row in task_rows])),
            "xyz_direction_cosine": float(np.mean([row["metrics"]["xyz_direction_cosine"] for row in task_rows])),
            "gripper_abs_error": float(np.mean([row["metrics"]["gripper_abs_error"] for row in task_rows])),
        }
    return {"num_samples": len(rows), "overall": summary, "by_task": by_task}


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    cfg, vla, processor, proprio_projector, action_head, depth_encoder = load_policy(args)
    rng = np.random.default_rng(args.seed)
    tasks = set(parse_csv(args.tasks)) if args.tasks else None
    depth_bank = None
    compare_depth_mode = str(args.compare_depth_mode or "").strip()
    depth_modes_needing_bank = [args.depth_mode]
    if compare_depth_mode:
        depth_modes_needing_bank.append(compare_depth_mode)
    if args.use_depth and any(needs_depth_bank(mode) for mode in depth_modes_needing_bank):
        depth_bank = collect_depth_bank(
            args.data_dir,
            tasks,
            max_bank_samples=args.depth_corrupt_bank_size,
            stride=args.depth_corrupt_bank_stride,
        )
        print(f"[depth-corrupt] collected {len(depth_bank)} cross-sample depth entries")
    rows: list[dict[str, Any]] = []

    for file_path in list_hdf5_files(args.data_dir, tasks):
        with h5py.File(file_path, "r") as f:
            file_task = read_hdf5_string(f.attrs.get("task_name", ""), default=file_path.stem)
            task_count = 0
            for demo_key in iter_demo_keys(f):
                demo = f["data"][demo_key]
                instruction = read_hdf5_string(
                    demo.attrs.get("language_instruction", None),
                    default=file_task.replace("_", " "),
                )
                target_actions = demo["rlbench_delta_action"] if "rlbench_delta_action" in demo else demo["actions"]
                for t in sample_indices(int(target_actions.shape[0]), args.stride):
                    obs = build_obs(demo, t, args.use_depth, args.depth_mode, rng, depth_bank=depth_bank)
                    pred_chunk = get_vla_action(
                        cfg,
                        vla,
                        processor,
                        obs,
                        instruction,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        depth_encoder=depth_encoder,
                    )
                    depth_debug = extract_depth_debug(action_head)
                    pred = np.asarray(pred_chunk[0], dtype=np.float32)
                    target = np.asarray(target_actions[t], dtype=np.float32)
                    aux_label = compute_aux_xyz_label(demo, t, args.aux_target, args.aux_future_horizon)
                    metrics = action_metrics(pred, target)
                    metrics.update(depth_debug_scalar_metrics(depth_debug))
                    metrics.update(aux_label_depth_point_metrics(depth_debug, aux_label))
                    if args.use_depth and compare_depth_mode:
                        compare_obs = build_obs(
                            demo,
                            t,
                            args.use_depth,
                            compare_depth_mode,
                            rng,
                            depth_bank=depth_bank,
                        )
                        compare_pred_chunk = get_vla_action(
                            cfg,
                            vla,
                            processor,
                            compare_obs,
                            instruction,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            depth_encoder=depth_encoder,
                        )
                        compare_debug = extract_depth_debug(action_head)
                        compare_pred = np.asarray(compare_pred_chunk[0], dtype=np.float32)
                        metrics.update(paired_action_delta_metrics(pred, compare_pred))
                        metrics.update(paired_depth_debug_metrics(depth_debug, compare_debug))
                        metrics.update(aux_label_depth_point_metrics(compare_debug, aux_label, prefix="compare_"))
                        if (
                            "depth_point_to_aux_label_l2" in metrics
                            and "compare_depth_point_to_aux_label_l2" in metrics
                        ):
                            metrics["paired_depth_point_aux_label_l2_advantage"] = (
                                metrics["compare_depth_point_to_aux_label_l2"]
                                - metrics["depth_point_to_aux_label_l2"]
                            )
                    rows.append(
                        {
                            "file": str(file_path),
                            "task": file_task,
                            "demo": demo_key,
                            "t": int(t),
                            "metrics": metrics,
                        }
                    )
                    task_count += 1
                    if args.max_samples and len(rows) >= args.max_samples:
                        return finish(args, rows)
                    if args.max_samples_per_task and task_count >= args.max_samples_per_task:
                        break
                if args.max_samples_per_task and task_count >= args.max_samples_per_task:
                    break
    return finish(args, rows)


def finish(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "use_depth": bool(args.use_depth),
        "depth_mode": args.depth_mode if args.use_depth else "rgb_only",
        "compare_depth_mode": args.compare_depth_mode if args.use_depth and args.compare_depth_mode else "",
        "stride": args.stride,
        **summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["overall"], indent=2))
    print(f"[done] wrote {args.output}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned checkpoint/run directory.")
    parser.add_argument("--base_model_checkpoint", default="", help="Base OpenVLA checkpoint for processor/model code.")
    parser.add_argument("--data_dir", required=True, type=Path, help="Converted RLBench HDF5 directory.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--unnorm_key", required=True)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_samples_per_task", type=int, default=0)
    parser.add_argument("--stride", type=int, default=20)
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
        "--compare_depth_mode",
        choices=("", "normal", "null", "shuffle", "shuffle_samples", "cross_sample", "replace_from_other_episode"),
        default="",
        help="Optional second depth mode predicted on the same sample to measure paired action deltas.",
    )
    parser.add_argument("--depth_corrupt_bank_size", type=int, default=2048)
    parser.add_argument("--depth_corrupt_bank_stride", type=int, default=10)
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
    parser.add_argument("--aux_target", default="")
    parser.add_argument("--aux_output_dim", type=int, default=8)
    parser.add_argument("--aux_future_horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[warn] CUDA is not available; model diagnostic will run on CPU and may be slow.")
    run_diagnostic(args)


if __name__ == "__main__":
    main()
