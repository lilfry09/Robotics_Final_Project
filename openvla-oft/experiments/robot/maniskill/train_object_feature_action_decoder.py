#!/usr/bin/env python3
"""Object-centric feature action decoder for ManiSkill PickCube.

This is a compact final pilot: instead of asking a tiny PointNet to discover
the cube from raw point clouds, extract the cube center from pointcloud
segmentation and train an MLP on explicit 3D task features.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_pointcloud_action_decoder import build_cross_source_indices, natural_key, require_dataset, safe_std


@dataclass
class ObjectFeatureData:
    cube_centers: np.ndarray
    cube_valid: np.ndarray
    tcp: np.ndarray
    goal: np.ndarray
    grasped: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray
    episode_ids: np.ndarray
    cross_source_indices: np.ndarray


@dataclass
class FeatureNormalizers:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


class ObjectFeatureActionDecoder(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def cube_center_from_arrays(xyzw: np.ndarray, segmentation: np.ndarray, cube_seg_id: int) -> tuple[np.ndarray, float]:
    xyz = np.asarray(xyzw[..., :3], dtype=np.float32)
    seg = np.asarray(segmentation).reshape(-1)
    pts = xyz[seg == cube_seg_id]
    pts = pts[np.all(np.isfinite(pts), axis=-1)]
    if len(pts) < 3:
        return np.zeros(3, dtype=np.float32), 0.0
    return pts.mean(axis=0).astype(np.float32), 1.0


def make_feature(
    cube_center: np.ndarray,
    cube_valid: float,
    tcp: np.ndarray,
    goal: np.ndarray,
    grasped: float,
    proprio: np.ndarray,
) -> np.ndarray:
    if cube_valid < 0.5:
        cube_center = np.zeros(3, dtype=np.float32)
        tcp_to_cube = np.zeros(3, dtype=np.float32)
        cube_to_goal = np.zeros(3, dtype=np.float32)
        dist_tcp_cube = 0.0
        dist_cube_goal = 0.0
    else:
        tcp_to_cube = cube_center - tcp
        cube_to_goal = goal - cube_center
        dist_tcp_cube = float(np.linalg.norm(tcp_to_cube))
        dist_cube_goal = float(np.linalg.norm(cube_to_goal))
    tcp_to_goal = goal - tcp
    dist_tcp_goal = float(np.linalg.norm(tcp_to_goal))
    return np.concatenate(
        [
            np.asarray([cube_valid], dtype=np.float32),
            cube_center.astype(np.float32),
            tcp.astype(np.float32),
            goal.astype(np.float32),
            np.asarray([grasped], dtype=np.float32),
            tcp_to_cube.astype(np.float32),
            cube_to_goal.astype(np.float32),
            tcp_to_goal.astype(np.float32),
            np.asarray([dist_tcp_cube, dist_cube_goal, dist_tcp_goal], dtype=np.float32),
            proprio.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def features_for_mode(data: ObjectFeatureData, indices: np.ndarray, mode: str) -> np.ndarray:
    if mode == "normal":
        cube = data.cube_centers[indices]
        valid = data.cube_valid[indices]
    elif mode == "null":
        cube = np.zeros_like(data.cube_centers[indices])
        valid = np.zeros_like(data.cube_valid[indices])
    elif mode == "cross_sample":
        source = data.cross_source_indices[indices]
        cube = data.cube_centers[source]
        valid = data.cube_valid[source]
    else:
        raise ValueError(f"unknown mode {mode}")
    return np.stack(
        [
            make_feature(cube_i, float(valid_i), tcp_i, goal_i, float(grasped_i), proprio_i)
            for cube_i, valid_i, tcp_i, goal_i, grasped_i, proprio_i in zip(
                cube, valid, data.tcp[indices], data.goal[indices], data.grasped[indices], data.proprio[indices]
            )
        ],
        axis=0,
    )


def load_data(path: Path, cube_seg_id: int, max_samples: int | None, seed: int) -> ObjectFeatureData:
    cube_centers: list[np.ndarray] = []
    cube_valid: list[float] = []
    tcp: list[np.ndarray] = []
    goal: list[np.ndarray] = []
    grasped: list[float] = []
    proprio: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_ids: list[int] = []

    with h5py.File(path, "r") as h5:
        names = sorted([name for name in h5.keys() if name.startswith(("traj_", "episode_"))], key=natural_key)
        if not names:
            raise ValueError(f"no traj_* or episode_* groups in {path}")
        for episode_idx, name in enumerate(names):
            group = h5[name]
            if not isinstance(group, h5py.Group):
                continue
            xyzw_ds = require_dataset(group, "obs/pointcloud/xyzw")
            seg_ds = require_dataset(group, "obs/pointcloud/segmentation")
            tcp_ds = require_dataset(group, "obs/extra/tcp_pose")
            goal_ds = require_dataset(group, "obs/extra/goal_pos")
            grasped_ds = require_dataset(group, "obs/extra/is_grasped")
            qpos_ds = require_dataset(group, "obs/agent/qpos")
            qvel_ds = require_dataset(group, "obs/agent/qvel")
            action_ds = require_dataset(group, "actions")
            length = min(
                len(action_ds),
                len(xyzw_ds),
                len(seg_ds),
                len(tcp_ds),
                len(goal_ds),
                len(grasped_ds),
                len(qpos_ds),
                len(qvel_ds),
            )
            for step_idx in range(length):
                center, valid = cube_center_from_arrays(
                    np.asarray(xyzw_ds[step_idx]), np.asarray(seg_ds[step_idx]), cube_seg_id
                )
                qpos = np.asarray(qpos_ds[step_idx], dtype=np.float32).reshape(-1)
                qvel = np.asarray(qvel_ds[step_idx], dtype=np.float32).reshape(-1)
                cube_centers.append(center)
                cube_valid.append(valid)
                tcp.append(np.asarray(tcp_ds[step_idx], dtype=np.float32).reshape(-1)[:3])
                goal.append(np.asarray(goal_ds[step_idx], dtype=np.float32).reshape(-1)[:3])
                grasped.append(float(np.asarray(grasped_ds[step_idx]).reshape(-1)[0]))
                proprio.append(np.concatenate([qpos, qvel], axis=0).astype(np.float32))
                actions.append(np.asarray(action_ds[step_idx], dtype=np.float32).reshape(-1))
                episode_ids.append(episode_idx)
                if max_samples is not None and len(actions) >= max_samples:
                    break
            if max_samples is not None and len(actions) >= max_samples:
                break
    if not actions:
        raise ValueError(f"no samples loaded from {path}")
    episode_arr = np.asarray(episode_ids, dtype=np.int64)
    return ObjectFeatureData(
        cube_centers=np.stack(cube_centers, axis=0),
        cube_valid=np.asarray(cube_valid, dtype=np.float32),
        tcp=np.stack(tcp, axis=0),
        goal=np.stack(goal, axis=0),
        grasped=np.asarray(grasped, dtype=np.float32),
        proprio=np.stack(proprio, axis=0),
        actions=np.stack(actions, axis=0),
        episode_ids=episode_arr,
        cross_source_indices=build_cross_source_indices(episode_arr, seed + 503),
    )


def split_by_episode(episode_ids: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_eps = np.unique(episode_ids)
    rng.shuffle(unique_eps)
    val_count = max(1, int(round(len(unique_eps) * val_fraction)))
    if len(unique_eps) > 1:
        val_count = min(val_count, len(unique_eps) - 1)
    val_eps = set(unique_eps[:val_count].tolist())
    val_idx = np.asarray([idx for idx, ep in enumerate(episode_ids) if int(ep) in val_eps], dtype=np.int64)
    train_idx = np.asarray([idx for idx, ep in enumerate(episode_ids) if int(ep) not in val_eps], dtype=np.int64)
    return train_idx, val_idx


def fit_normalizers(data: ObjectFeatureData, train_idx: np.ndarray) -> FeatureNormalizers:
    train_features = features_for_mode(data, train_idx, "normal")
    train_actions = data.actions[train_idx]
    return FeatureNormalizers(
        feature_mean=train_features.mean(axis=0).astype(np.float32),
        feature_std=safe_std(train_features, axis=0),
        action_mean=train_actions.mean(axis=0).astype(np.float32),
        action_std=safe_std(train_actions, axis=0),
    )


def normalize_features(features: np.ndarray, norm: FeatureNormalizers) -> np.ndarray:
    return ((features - norm.feature_mean.reshape(1, -1)) / norm.feature_std.reshape(1, -1)).astype(np.float32)


def normalize_actions(actions: np.ndarray, norm: FeatureNormalizers) -> np.ndarray:
    return ((actions - norm.action_mean.reshape(1, -1)) / norm.action_std.reshape(1, -1)).astype(np.float32)


def denormalize_actions(actions: np.ndarray, norm: FeatureNormalizers) -> np.ndarray:
    return (actions * norm.action_std.reshape(1, -1) + norm.action_mean.reshape(1, -1)).astype(np.float32)


def batch_indices(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


@torch.no_grad()
def evaluate_mode(
    model: nn.Module,
    data: ObjectFeatureData,
    indices: np.ndarray,
    norm: FeatureNormalizers,
    mode: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    norm_losses: list[float] = []
    for idx in batch_indices(indices, batch_size):
        features = torch.as_tensor(normalize_features(features_for_mode(data, idx, mode), norm), device=device)
        target = torch.as_tensor(normalize_actions(data.actions[idx], norm), device=device)
        pred = model(features)
        norm_losses.append(float(F.mse_loss(pred, target).detach().cpu()))
        preds.append(denormalize_actions(pred.detach().cpu().numpy(), norm))
        targets.append(data.actions[idx])
    pred_raw = np.concatenate(preds, axis=0)
    target_raw = np.concatenate(targets, axis=0)
    err = pred_raw - target_raw
    return {
        "norm_mse": float(np.mean(norm_losses)),
        "raw_mse": float(np.mean(err**2)),
        "raw_rmse": float(np.sqrt(np.mean(err**2))),
        "raw_l1": float(np.mean(np.abs(err))),
    }


@torch.no_grad()
def paired_delta(
    model: nn.Module,
    data: ObjectFeatureData,
    indices: np.ndarray,
    norm: FeatureNormalizers,
    other_mode: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    deltas: list[np.ndarray] = []
    for idx in batch_indices(indices, batch_size):
        normal = torch.as_tensor(normalize_features(features_for_mode(data, idx, "normal"), norm), device=device)
        other = torch.as_tensor(normalize_features(features_for_mode(data, idx, other_mode), norm), device=device)
        pred_normal = denormalize_actions(model(normal).detach().cpu().numpy(), norm)
        pred_other = denormalize_actions(model(other).detach().cpu().numpy(), norm)
        deltas.append(pred_normal - pred_other)
    delta = np.concatenate(deltas, axis=0)
    return {
        f"paired_normal_vs_{other_mode}_l1": float(np.mean(np.abs(delta))),
        f"paired_normal_vs_{other_mode}_l2": float(np.mean(np.linalg.norm(delta, axis=-1))),
        f"paired_normal_vs_{other_mode}_max_l2": float(np.max(np.linalg.norm(delta, axis=-1))),
    }


def normalizers_to_tensors(norm: FeatureNormalizers) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value) for key, value in norm.__dict__.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load_data(args.input, args.cube_seg_id, args.max_samples, args.seed)
    train_idx, val_idx = split_by_episode(data.episode_ids, args.val_fraction, args.seed)
    norm = fit_normalizers(data, train_idx)
    feature_dim = int(features_for_mode(data, train_idx[:1], "normal").shape[-1])
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ObjectFeatureActionDecoder(feature_dim, data.actions.shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 17)
    for step in range(1, args.steps + 1):
        idx = rng.choice(train_idx, size=args.batch_size, replace=len(train_idx) < args.batch_size)
        features = torch.as_tensor(normalize_features(features_for_mode(data, idx, "normal"), norm), device=device)
        target = torch.as_tensor(normalize_actions(data.actions[idx], norm), device=device)
        pred = model(features)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if args.log_every and step % args.log_every == 0:
            print(f"step {step:04d} train_norm_mse={float(loss.detach().cpu()):.6f}")

    metrics = {
        mode: evaluate_mode(model, data, val_idx, norm, mode, args.eval_batch_size, device)
        for mode in ("normal", "null", "cross_sample")
    }
    deltas = {}
    deltas.update(paired_delta(model, data, val_idx, norm, "null", args.eval_batch_size, device))
    deltas.update(paired_delta(model, data, val_idx, norm, "cross_sample", args.eval_batch_size, device))
    normal_mse = metrics["normal"]["raw_mse"]
    null_mse = metrics["null"]["raw_mse"]
    cross_mse = metrics["cross_sample"]["raw_mse"]
    cross_delta = deltas["paired_normal_vs_cross_sample_l2"]
    gate_pass = bool(normal_mse < null_mse and normal_mse < cross_mse and cross_delta >= args.min_paired_delta)
    result: dict[str, Any] = {
        "input": str(args.input),
        "num_samples": int(len(data.actions)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_episodes": int(len(np.unique(data.episode_ids))),
        "feature_dim": feature_dim,
        "action_dim": int(data.actions.shape[-1]),
        "steps": int(args.steps),
        "cube_seg_id": int(args.cube_seg_id),
        "metrics": metrics,
        "paired_deltas": deltas,
        "gate": {
            "passed": gate_pass,
            "normal_raw_mse_lt_null": bool(normal_mse < null_mse),
            "normal_raw_mse_lt_cross_sample": bool(normal_mse < cross_mse),
            "paired_normal_vs_cross_l2": float(cross_delta),
            "interpretation": (
                "GO: object-centric normal geometry improves action prediction."
                if gate_pass
                else "NO-GO: object-centric features do not beat null/cross."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.checkpoint_output is not None:
        args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "normalizers": normalizers_to_tensors(norm),
                "model_config": {
                    "feature_dim": feature_dim,
                    "action_dim": int(data.actions.shape[-1]),
                    "hidden_dim": int(args.hidden_dim),
                    "cube_seg_id": int(args.cube_seg_id),
                },
                "train_config": {"input": str(args.input), "steps": int(args.steps), "seed": int(args.seed)},
                "result": result,
            },
            args.checkpoint_output,
        )
        print(f"wrote {args.checkpoint_output}")
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint_output", type=Path, default=None)
    parser.add_argument("--cube_seg_id", type=int, default=18)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--min_paired_delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=1000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
