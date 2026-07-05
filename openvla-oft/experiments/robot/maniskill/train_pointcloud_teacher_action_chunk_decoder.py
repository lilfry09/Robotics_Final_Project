#!/usr/bin/env python3
"""Train a cropped-pointcloud teacher action-chunk decoder for PickCube."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_pointcloud_action_decoder import build_cross_source_indices, safe_std
from train_teacher_object_feature_decoder import batch_indices, split_by_episode


@dataclass
class ChunkTeacherData:
    points: np.ndarray
    task_features: np.ndarray
    cube_centers: np.ndarray
    cube_valid: np.ndarray
    phase_labels: np.ndarray
    action_chunks: np.ndarray
    episode_ids: np.ndarray
    cross_source_indices: np.ndarray
    phase_names: list[str]


@dataclass
class ChunkNormalizers:
    point_mean: np.ndarray
    point_std: np.ndarray
    task_mean: np.ndarray
    task_std: np.ndarray
    cube_mean: np.ndarray
    cube_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


class PointNetTeacherActionChunkDecoder(nn.Module):
    def __init__(
        self,
        point_dim: int,
        task_dim: int,
        action_dim: int,
        num_phases: int,
        chunk_horizon: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_horizon = chunk_horizon
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.task_mlp = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.phase_head = nn.Linear(hidden_dim, num_phases)
        self.cube_head = nn.Linear(hidden_dim, 4)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + num_phases + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_horizon * action_dim),
        )

    def forward(
        self,
        points: torch.Tensor,
        task_features: torch.Tensor,
        phase_for_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        point_features = self.point_mlp(points).max(dim=1).values
        task_features = self.task_mlp(task_features)
        hidden = self.trunk(torch.cat([point_features, task_features], dim=-1))
        phase_logits = self.phase_head(hidden)
        cube_pred = self.cube_head(hidden)
        if phase_for_action is None:
            phase_for_action = torch.softmax(phase_logits, dim=-1)
        flat = self.action_head(torch.cat([hidden, phase_for_action, cube_pred[:, :3]], dim=-1))
        return flat.reshape(points.shape[0], self.chunk_horizon, self.action_dim), phase_logits, cube_pred


def valid_chunk_starts(episode_ids: np.ndarray, horizon: int) -> np.ndarray:
    starts: list[int] = []
    for idx in range(0, len(episode_ids) - horizon + 1):
        if np.all(episode_ids[idx : idx + horizon] == episode_ids[idx]):
            starts.append(idx)
    return np.asarray(starts, dtype=np.int64)


def load_data(path: Path, horizon: int, seed: int) -> ChunkTeacherData:
    data = np.load(path)
    starts = valid_chunk_starts(np.asarray(data["episode_ids"], dtype=np.int64), horizon)
    if len(starts) == 0:
        raise ValueError(f"no valid {horizon}-step chunks in {path}")
    actions = np.asarray(data["actions"], dtype=np.float32)
    action_chunks = np.stack([actions[start : start + horizon] for start in starts], axis=0).astype(np.float32)
    episode_ids = np.asarray(data["episode_ids"], dtype=np.int64)[starts]
    phase_names = [str(item) for item in np.asarray(data["phase_names"], dtype=str).reshape(-1).tolist()]
    return ChunkTeacherData(
        points=np.asarray(data["points"], dtype=np.float32)[starts],
        task_features=np.asarray(data["task_features"], dtype=np.float32)[starts],
        cube_centers=np.asarray(data["cube_centers"], dtype=np.float32)[starts],
        cube_valid=np.asarray(data["cube_valid"], dtype=np.float32)[starts],
        phase_labels=np.asarray(data["phase_labels"], dtype=np.int64)[starts],
        action_chunks=action_chunks,
        episode_ids=episode_ids,
        cross_source_indices=build_cross_source_indices(episode_ids, seed + 1301),
        phase_names=phase_names,
    )


def fit_normalizers(data: ChunkTeacherData, train_idx: np.ndarray) -> ChunkNormalizers:
    train_points = data.points[train_idx]
    train_actions = data.action_chunks[train_idx].reshape(-1, data.action_chunks.shape[-1])
    return ChunkNormalizers(
        point_mean=train_points.reshape(-1, train_points.shape[-1]).mean(axis=0).astype(np.float32),
        point_std=safe_std(train_points.reshape(-1, train_points.shape[-1]), axis=0),
        task_mean=data.task_features[train_idx].mean(axis=0).astype(np.float32),
        task_std=safe_std(data.task_features[train_idx], axis=0),
        cube_mean=data.cube_centers[train_idx].mean(axis=0).astype(np.float32),
        cube_std=safe_std(data.cube_centers[train_idx], axis=0),
        action_mean=train_actions.mean(axis=0).astype(np.float32),
        action_std=safe_std(train_actions, axis=0),
    )


def normalize_points(points: np.ndarray, norm: ChunkNormalizers) -> np.ndarray:
    return ((points - norm.point_mean.reshape(1, 1, -1)) / norm.point_std.reshape(1, 1, -1)).astype(np.float32)


def normalize_task(task: np.ndarray, norm: ChunkNormalizers) -> np.ndarray:
    return ((task - norm.task_mean.reshape(1, -1)) / norm.task_std.reshape(1, -1)).astype(np.float32)


def normalize_cube(cube: np.ndarray, norm: ChunkNormalizers) -> np.ndarray:
    return ((cube - norm.cube_mean.reshape(1, -1)) / norm.cube_std.reshape(1, -1)).astype(np.float32)


def denormalize_cube(cube: np.ndarray, norm: ChunkNormalizers) -> np.ndarray:
    return (cube * norm.cube_std.reshape(1, -1) + norm.cube_mean.reshape(1, -1)).astype(np.float32)


def normalize_chunks(chunks: np.ndarray, norm: ChunkNormalizers) -> np.ndarray:
    return ((chunks - norm.action_mean.reshape(1, 1, -1)) / norm.action_std.reshape(1, 1, -1)).astype(np.float32)


def denormalize_chunks(chunks: np.ndarray, norm: ChunkNormalizers) -> np.ndarray:
    return (chunks * norm.action_std.reshape(1, 1, -1) + norm.action_mean.reshape(1, 1, -1)).astype(np.float32)


def make_points_for_mode(data: ChunkTeacherData, indices: np.ndarray, mode: str, norm: ChunkNormalizers) -> np.ndarray:
    if mode == "normal":
        raw = data.points[indices]
    elif mode == "null":
        raw = np.zeros_like(data.points[indices])
    elif mode == "cross_sample":
        raw = data.points[data.cross_source_indices[indices]]
    else:
        raise ValueError(f"unknown mode: {mode}")
    return normalize_points(raw, norm)


def normalizers_to_tensors(norm: ChunkNormalizers) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value) for key, value in norm.__dict__.items()}


def class_weights(labels: np.ndarray, train_idx: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels[train_idx], minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (float(nonzero.sum()) * counts[nonzero])
    return weights


@torch.no_grad()
def evaluate_mode(
    model: nn.Module,
    data: ChunkTeacherData,
    indices: np.ndarray,
    norm: ChunkNormalizers,
    mode: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    phase_preds: list[np.ndarray] = []
    cube_errors: list[np.ndarray] = []
    losses: list[float] = []
    for idx in batch_indices(indices, batch_size):
        points = torch.as_tensor(make_points_for_mode(data, idx, mode, norm), device=device)
        task = torch.as_tensor(normalize_task(data.task_features[idx], norm), device=device)
        target = torch.as_tensor(normalize_chunks(data.action_chunks[idx], norm), device=device)
        pred, logits, cube_pred = model(points, task, None)
        losses.append(float(F.mse_loss(pred, target).detach().cpu()))
        preds.append(denormalize_chunks(pred.detach().cpu().numpy(), norm))
        targets.append(data.action_chunks[idx])
        phase_preds.append(logits.argmax(dim=-1).detach().cpu().numpy())
        cube_errors.append(denormalize_cube(cube_pred[:, :3].detach().cpu().numpy(), norm) - data.cube_centers[idx])
    pred_raw = np.concatenate(preds, axis=0)
    target_raw = np.concatenate(targets, axis=0)
    err = pred_raw - target_raw
    cube_err = np.concatenate(cube_errors, axis=0)
    phase_pred = np.concatenate(phase_preds, axis=0)
    return {
        "norm_mse": float(np.mean(losses)),
        "raw_mse": float(np.mean(err**2)),
        "raw_rmse": float(np.sqrt(np.mean(err**2))),
        "raw_l1": float(np.mean(np.abs(err))),
        "phase_accuracy": float(np.mean(phase_pred == data.phase_labels[indices])),
        "cube_xyz_rmse": float(np.sqrt(np.mean(cube_err**2))),
    }


@torch.no_grad()
def paired_delta(
    model: nn.Module,
    data: ChunkTeacherData,
    indices: np.ndarray,
    norm: ChunkNormalizers,
    other_mode: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    deltas: list[np.ndarray] = []
    for idx in batch_indices(indices, batch_size):
        task = torch.as_tensor(normalize_task(data.task_features[idx], norm), device=device)
        normal = torch.as_tensor(make_points_for_mode(data, idx, "normal", norm), device=device)
        other = torch.as_tensor(make_points_for_mode(data, idx, other_mode, norm), device=device)
        pred_normal, _, _ = model(normal, task, None)
        pred_other, _, _ = model(other, task, None)
        deltas.append(
            denormalize_chunks(pred_normal.detach().cpu().numpy(), norm)
            - denormalize_chunks(pred_other.detach().cpu().numpy(), norm)
        )
    delta = np.concatenate(deltas, axis=0)
    step_l2 = np.linalg.norm(delta, axis=-1)
    return {
        f"paired_normal_vs_{other_mode}_l1": float(np.mean(np.abs(delta))),
        f"paired_normal_vs_{other_mode}_step_l2": float(np.mean(step_l2)),
        f"paired_normal_vs_{other_mode}_chunk_l2": float(np.mean(np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=-1))),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load_data(args.input, args.chunk_horizon, args.seed)
    train_idx, val_idx = split_by_episode(data.episode_ids, args.val_fraction, args.seed)
    norm = fit_normalizers(data, train_idx)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PointNetTeacherActionChunkDecoder(
        point_dim=data.points.shape[-1],
        task_dim=data.task_features.shape[-1],
        action_dim=data.action_chunks.shape[-1],
        num_phases=len(data.phase_names),
        chunk_horizon=args.chunk_horizon,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = torch.as_tensor(class_weights(data.phase_labels, train_idx, len(data.phase_names)), device=device)
    rng = np.random.default_rng(args.seed + 37)
    for step in range(1, args.steps + 1):
        idx = rng.choice(train_idx, size=args.batch_size, replace=len(train_idx) < args.batch_size)
        points = torch.as_tensor(make_points_for_mode(data, idx, "normal", norm), device=device)
        task = torch.as_tensor(normalize_task(data.task_features[idx], norm), device=device)
        target = torch.as_tensor(normalize_chunks(data.action_chunks[idx], norm), device=device)
        cube_target = torch.as_tensor(normalize_cube(data.cube_centers[idx], norm), device=device)
        cube_valid = torch.as_tensor(data.cube_valid[idx], device=device)
        phase_labels = torch.as_tensor(data.phase_labels[idx], dtype=torch.long, device=device)
        phase_for_action = F.one_hot(phase_labels, num_classes=len(data.phase_names)).float()
        pred, logits, cube_pred = model(points, task, phase_for_action)
        action_loss = F.mse_loss(pred, target)
        phase_loss = F.cross_entropy(logits, phase_labels, weight=weights, label_smoothing=args.label_smoothing)
        valid_mask = cube_valid.reshape(-1, 1)
        cube_xyz_loss = ((cube_pred[:, :3] - cube_target) ** 2 * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        cube_valid_loss = F.binary_cross_entropy_with_logits(cube_pred[:, 3], cube_valid)
        loss = action_loss + args.phase_loss_weight * phase_loss + args.cube_loss_weight * (cube_xyz_loss + 0.1 * cube_valid_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if args.log_every and step % args.log_every == 0:
            print(f"step {step:04d} loss={float(loss.detach().cpu()):.6f} action={float(action_loss.detach().cpu()):.6f}")

    metrics = {
        mode: evaluate_mode(model, data, val_idx, norm, mode, args.eval_batch_size, device)
        for mode in ("normal", "null", "cross_sample")
    }
    deltas: dict[str, float] = {}
    deltas.update(paired_delta(model, data, val_idx, norm, "null", args.eval_batch_size, device))
    deltas.update(paired_delta(model, data, val_idx, norm, "cross_sample", args.eval_batch_size, device))
    normal_mse = metrics["normal"]["raw_mse"]
    null_mse = metrics["null"]["raw_mse"]
    cross_mse = metrics["cross_sample"]["raw_mse"]
    cross_delta = deltas["paired_normal_vs_cross_sample_step_l2"]
    gate_pass = bool(normal_mse < null_mse and normal_mse < cross_mse and cross_delta >= args.min_paired_delta)
    result: dict[str, Any] = {
        "input": str(args.input),
        "num_samples": int(len(data.action_chunks)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_episodes": int(len(np.unique(data.episode_ids))),
        "chunk_horizon": int(args.chunk_horizon),
        "num_points": int(data.points.shape[1]),
        "point_dim": int(data.points.shape[-1]),
        "task_dim": int(data.task_features.shape[-1]),
        "action_dim": int(data.action_chunks.shape[-1]),
        "phase_names": data.phase_names,
        "steps": int(args.steps),
        "metrics": metrics,
        "paired_deltas": deltas,
        "gate": {
            "passed": gate_pass,
            "normal_raw_mse_lt_null": bool(normal_mse < null_mse),
            "normal_raw_mse_lt_cross_sample": bool(normal_mse < cross_mse),
            "paired_normal_vs_cross_step_l2": float(cross_delta),
            "min_paired_delta": float(args.min_paired_delta),
        },
        "note": "Teacher action-chunk decoder from cropped raw pointcloud/task features.",
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
                    "point_dim": int(data.points.shape[-1]),
                    "task_dim": int(data.task_features.shape[-1]),
                    "action_dim": int(data.action_chunks.shape[-1]),
                    "num_phases": int(len(data.phase_names)),
                    "phase_names": data.phase_names,
                    "chunk_horizon": int(args.chunk_horizon),
                    "hidden_dim": int(args.hidden_dim),
                    "num_points": int(data.points.shape[1]),
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
    parser.add_argument("--chunk_horizon", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--phase_loss_weight", type=float, default=0.2)
    parser.add_argument("--cube_loss_weight", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--min_paired_delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=1000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
