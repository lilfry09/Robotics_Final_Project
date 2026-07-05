#!/usr/bin/env python3
"""Tiny point-cloud action chunk decoder for ManiSkill demos."""

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

from train_pointcloud_action_decoder import (
    Normalizers,
    build_cross_source_indices,
    denormalize_actions,
    natural_key,
    normalize_points,
    normalize_proprio,
    require_dataset,
    safe_std,
    select_points,
)


@dataclass
class ChunkData:
    points: np.ndarray
    proprio: np.ndarray
    action_chunks: np.ndarray
    episode_ids: np.ndarray
    cross_source_indices: np.ndarray
    source_steps: list[str]


class PointNetActionChunkDecoder(nn.Module):
    def __init__(
        self,
        point_dim: int,
        proprio_dim: int,
        action_dim: int,
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
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_horizon * action_dim),
        )

    def forward(self, points: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        point_features = self.point_mlp(points).max(dim=1).values
        proprio_features = self.proprio_mlp(proprio)
        flat = self.head(torch.cat([point_features, proprio_features], dim=-1))
        return flat.reshape(points.shape[0], self.chunk_horizon, self.action_dim)


def load_chunk_data(path: Path, num_points: int, chunk_horizon: int, max_samples: int | None, seed: int) -> ChunkData:
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    episode_ids: list[int] = []
    source_steps: list[str] = []

    with h5py.File(path, "r") as h5:
        group_names = sorted(
            [name for name in h5.keys() if name.startswith(("traj_", "episode_"))],
            key=natural_key,
        )
        if not group_names:
            raise ValueError(f"no traj_* or episode_* groups in {path}")

        for episode_idx, name in enumerate(group_names):
            group = h5[name]
            if not isinstance(group, h5py.Group):
                continue
            xyzw_ds = require_dataset(group, "obs/pointcloud/xyzw")
            qpos_ds = require_dataset(group, "obs/agent/qpos")
            qvel_ds = require_dataset(group, "obs/agent/qvel")
            action_ds = require_dataset(group, "actions")
            rgb_ds = group.get("obs/pointcloud/rgb")
            length = min(len(action_ds), len(xyzw_ds), len(qpos_ds), len(qvel_ds))
            if rgb_ds is not None:
                length = min(length, len(rgb_ds))
            if length < chunk_horizon:
                continue
            actions = np.asarray(action_ds[()], dtype=np.float32)
            for step_idx in range(length - chunk_horizon + 1):
                points.append(
                    select_points(
                        np.asarray(xyzw_ds[step_idx]),
                        np.asarray(rgb_ds[step_idx]) if rgb_ds is not None else None,
                        num_points,
                        rng,
                    )
                )
                qpos = np.asarray(qpos_ds[step_idx], dtype=np.float32).reshape(-1)
                qvel = np.asarray(qvel_ds[step_idx], dtype=np.float32).reshape(-1)
                proprio.append(np.concatenate([qpos, qvel], axis=0).astype(np.float32))
                action_chunks.append(actions[step_idx : step_idx + chunk_horizon])
                episode_ids.append(episode_idx)
                source_steps.append(f"{name}:{step_idx}")
                if max_samples is not None and len(action_chunks) >= max_samples:
                    break
            if max_samples is not None and len(action_chunks) >= max_samples:
                break

    if not action_chunks:
        raise ValueError(f"no chunk samples loaded from {path}")
    episode_id_arr = np.asarray(episode_ids, dtype=np.int64)
    return ChunkData(
        points=np.stack(points, axis=0),
        proprio=np.stack(proprio, axis=0),
        action_chunks=np.stack(action_chunks, axis=0),
        episode_ids=episode_id_arr,
        cross_source_indices=build_cross_source_indices(episode_id_arr, seed + 211),
        source_steps=source_steps,
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


def fit_normalizers(data: ChunkData, train_idx: np.ndarray) -> Normalizers:
    train_points = data.points[train_idx]
    train_actions = data.action_chunks[train_idx].reshape(-1, data.action_chunks.shape[-1])
    return Normalizers(
        point_mean=train_points.reshape(-1, train_points.shape[-1]).mean(axis=0).astype(np.float32),
        point_std=safe_std(train_points.reshape(-1, train_points.shape[-1]), axis=0),
        proprio_mean=data.proprio[train_idx].mean(axis=0).astype(np.float32),
        proprio_std=safe_std(data.proprio[train_idx], axis=0),
        action_mean=train_actions.mean(axis=0).astype(np.float32),
        action_std=safe_std(train_actions, axis=0),
    )


def normalize_chunks(chunks: np.ndarray, norm: Normalizers) -> np.ndarray:
    return ((chunks - norm.action_mean.reshape(1, 1, -1)) / norm.action_std.reshape(1, 1, -1)).astype(np.float32)


def denormalize_chunks(chunks: np.ndarray, norm: Normalizers) -> np.ndarray:
    flat = chunks.reshape(-1, chunks.shape[-1])
    denorm = denormalize_actions(flat, norm)
    return denorm.reshape(chunks.shape).astype(np.float32)


def make_points_for_mode(data: ChunkData, indices: np.ndarray, mode: str, norm: Normalizers) -> np.ndarray:
    if mode == "normal":
        raw = data.points[indices]
    elif mode == "null":
        raw = np.zeros_like(data.points[indices])
    elif mode == "cross_sample":
        raw = data.points[data.cross_source_indices[indices]]
    else:
        raise ValueError(f"unknown point mode: {mode}")
    return normalize_points(raw, norm)


def batch_indices(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


@torch.no_grad()
def evaluate_mode(
    model: nn.Module,
    data: ChunkData,
    indices: np.ndarray,
    norm: Normalizers,
    mode: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    norm_losses: list[float] = []
    for idx in batch_indices(indices, batch_size):
        points = torch.as_tensor(make_points_for_mode(data, idx, mode, norm), device=device)
        proprio = torch.as_tensor(normalize_proprio(data.proprio[idx], norm), device=device)
        target = torch.as_tensor(normalize_chunks(data.action_chunks[idx], norm), device=device)
        pred = model(points, proprio)
        norm_losses.append(float(F.mse_loss(pred, target).detach().cpu()))
        preds.append(denormalize_chunks(pred.detach().cpu().numpy(), norm))
        targets.append(data.action_chunks[idx])
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
    data: ChunkData,
    indices: np.ndarray,
    norm: Normalizers,
    other_mode: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    deltas: list[np.ndarray] = []
    for idx in batch_indices(indices, batch_size):
        normal_points = torch.as_tensor(make_points_for_mode(data, idx, "normal", norm), device=device)
        other_points = torch.as_tensor(make_points_for_mode(data, idx, other_mode, norm), device=device)
        proprio = torch.as_tensor(normalize_proprio(data.proprio[idx], norm), device=device)
        normal_pred = denormalize_chunks(model(normal_points, proprio).detach().cpu().numpy(), norm)
        other_pred = denormalize_chunks(model(other_points, proprio).detach().cpu().numpy(), norm)
        deltas.append(normal_pred - other_pred)
    delta = np.concatenate(deltas, axis=0)
    step_l2 = np.linalg.norm(delta, axis=-1)
    return {
        f"paired_normal_vs_{other_mode}_l1": float(np.mean(np.abs(delta))),
        f"paired_normal_vs_{other_mode}_step_l2": float(np.mean(step_l2)),
        f"paired_normal_vs_{other_mode}_chunk_l2": float(np.mean(np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=-1))),
    }


def normalizers_to_tensors(norm: Normalizers) -> dict[str, torch.Tensor]:
    return {
        "point_mean": torch.as_tensor(norm.point_mean),
        "point_std": torch.as_tensor(norm.point_std),
        "proprio_mean": torch.as_tensor(norm.proprio_mean),
        "proprio_std": torch.as_tensor(norm.proprio_std),
        "action_mean": torch.as_tensor(norm.action_mean),
        "action_std": torch.as_tensor(norm.action_std),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = load_chunk_data(args.input, args.num_points, args.chunk_horizon, args.max_samples, args.seed)
    train_idx, val_idx = split_by_episode(data.episode_ids, args.val_fraction, args.seed)
    norm = fit_normalizers(data, train_idx)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PointNetActionChunkDecoder(
        point_dim=data.points.shape[-1],
        proprio_dim=data.proprio.shape[-1],
        action_dim=data.action_chunks.shape[-1],
        chunk_horizon=args.chunk_horizon,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 17)

    for step in range(1, args.steps + 1):
        replace = len(train_idx) < args.batch_size
        idx = rng.choice(train_idx, size=args.batch_size, replace=replace)
        points = torch.as_tensor(make_points_for_mode(data, idx, args.train_point_mode, norm), device=device)
        proprio = torch.as_tensor(normalize_proprio(data.proprio[idx], norm), device=device)
        target = torch.as_tensor(normalize_chunks(data.action_chunks[idx], norm), device=device)
        pred = model(points, proprio)
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
    cross_delta = deltas["paired_normal_vs_cross_sample_step_l2"]
    gate_pass = bool(normal_mse < null_mse and normal_mse < cross_mse and cross_delta >= args.min_paired_delta)
    result: dict[str, Any] = {
        "input": str(args.input),
        "num_samples": int(len(data.action_chunks)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_episodes": int(len(np.unique(data.episode_ids))),
        "num_points": int(args.num_points),
        "chunk_horizon": int(args.chunk_horizon),
        "action_dim": int(data.action_chunks.shape[-1]),
        "proprio_dim": int(data.proprio.shape[-1]),
        "steps": int(args.steps),
        "metrics": metrics,
        "paired_deltas": deltas,
        "gate": {
            "passed": gate_pass,
            "normal_raw_mse_lt_null": bool(normal_mse < null_mse),
            "normal_raw_mse_lt_cross_sample": bool(normal_mse < cross_mse),
            "paired_normal_vs_cross_step_l2": float(cross_delta),
            "interpretation": (
                "GO: normal pointcloud improves chunk prediction and corruptions change predictions."
                if gate_pass
                else "NO-GO: chunk decoder still lacks defensible normal pointcloud advantage."
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
                    "point_dim": int(data.points.shape[-1]),
                    "proprio_dim": int(data.proprio.shape[-1]),
                    "action_dim": int(data.action_chunks.shape[-1]),
                    "chunk_horizon": int(args.chunk_horizon),
                    "hidden_dim": int(args.hidden_dim),
                    "num_points": int(args.num_points),
                },
                "train_config": {
                    "input": str(args.input),
                    "train_point_mode": args.train_point_mode,
                    "steps": int(args.steps),
                    "seed": int(args.seed),
                },
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
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--chunk_horizon", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--train_point_mode", choices=["normal", "null", "cross_sample"], default="normal")
    parser.add_argument("--min_paired_delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=500)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
