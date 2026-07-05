#!/usr/bin/env python3
"""Tiny point-cloud action decoder gate for ManiSkill demos.

This is intentionally small. It answers one pre-rollout question:

    Does action prediction depend on real point-cloud content, or can proprio
    and shortcuts solve the loss equally well under null/cross-sample clouds?

The script trains on normal point clouds and evaluates the same checkpoint with
normal, zeroed, and cross-sample point clouds.
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


@dataclass
class LoadedData:
    points: np.ndarray
    proprio: np.ndarray
    actions: np.ndarray
    episode_ids: np.ndarray
    cross_source_indices: np.ndarray
    source_steps: list[str]


def natural_key(name: str) -> tuple[str, int | str]:
    prefix, _, suffix = name.rpartition("_")
    if suffix.isdigit():
        return prefix, int(suffix)
    return name, suffix


def require_dataset(group: h5py.Group, key: str) -> h5py.Dataset:
    if key not in group or not isinstance(group[key], h5py.Dataset):
        raise ValueError(f"missing dataset `{group.name}/{key}`")
    return group[key]


def select_points(
    xyzw: np.ndarray,
    rgb: np.ndarray | None,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    xyz = np.asarray(xyzw[..., :3], dtype=np.float32)
    finite = np.all(np.isfinite(xyz), axis=-1)
    if xyzw.shape[-1] >= 4:
        finite &= np.asarray(xyzw[..., 3] > 0.5)
    valid_idx = np.flatnonzero(finite)
    if valid_idx.size == 0:
        valid_idx = np.arange(xyz.shape[0])

    replace = valid_idx.size < num_points
    chosen = rng.choice(valid_idx, size=num_points, replace=replace)
    chosen_xyz = xyz[chosen]
    if rgb is None:
        chosen_rgb = np.zeros((num_points, 3), dtype=np.float32)
    else:
        chosen_rgb = np.asarray(rgb[chosen], dtype=np.float32)
        if chosen_rgb.max(initial=0.0) > 1.5:
            chosen_rgb = chosen_rgb / 255.0
    return np.concatenate([chosen_xyz, chosen_rgb], axis=-1).astype(np.float32)


def load_hdf5_samples(
    path: Path,
    num_points: int,
    max_samples: int | None,
    seed: int,
) -> LoadedData:
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    actions: list[np.ndarray] = []
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
            for step_idx in range(length):
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
                actions.append(np.asarray(action_ds[step_idx], dtype=np.float32).reshape(-1))
                episode_ids.append(episode_idx)
                source_steps.append(f"{name}:{step_idx}")
                if max_samples is not None and len(actions) >= max_samples:
                    break
            if max_samples is not None and len(actions) >= max_samples:
                break

    if not actions:
        raise ValueError(f"no samples loaded from {path}")
    episode_id_arr = np.asarray(episode_ids, dtype=np.int64)
    return LoadedData(
        points=np.stack(points, axis=0),
        proprio=np.stack(proprio, axis=0),
        actions=np.stack(actions, axis=0),
        episode_ids=episode_id_arr,
        cross_source_indices=build_cross_source_indices(episode_id_arr, seed + 101),
        source_steps=source_steps,
    )


def build_cross_source_indices(episode_ids: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(episode_ids), dtype=np.int64)
    cross = np.empty_like(all_indices)
    for idx, episode_id in enumerate(episode_ids):
        candidates = all_indices[episode_ids != episode_id]
        if candidates.size == 0:
            candidates = all_indices[all_indices != idx]
        if candidates.size == 0:
            candidates = all_indices
        cross[idx] = int(rng.choice(candidates))
    return cross


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
    if train_idx.size == 0:
        rng = np.random.default_rng(seed)
        all_idx = np.arange(len(episode_ids))
        rng.shuffle(all_idx)
        cut = max(1, int(round(len(all_idx) * (1.0 - val_fraction))))
        train_idx, val_idx = all_idx[:cut], all_idx[cut:]
    if val_idx.size == 0:
        val_idx = train_idx.copy()
    return train_idx, val_idx


class PointNetActionDecoder(nn.Module):
    def __init__(self, point_dim: int, proprio_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
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
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, points: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        point_features = self.point_mlp(points).max(dim=1).values
        proprio_features = self.proprio_mlp(proprio)
        return self.head(torch.cat([point_features, proprio_features], dim=-1))


def safe_std(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    std = values.std(axis=axis, keepdims=False).astype(np.float32)
    return np.maximum(std, 1e-6)


@dataclass
class Normalizers:
    point_mean: np.ndarray
    point_std: np.ndarray
    proprio_mean: np.ndarray
    proprio_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


def normalizers_to_tensors(norm: Normalizers) -> dict[str, torch.Tensor]:
    return {
        "point_mean": torch.as_tensor(norm.point_mean),
        "point_std": torch.as_tensor(norm.point_std),
        "proprio_mean": torch.as_tensor(norm.proprio_mean),
        "proprio_std": torch.as_tensor(norm.proprio_std),
        "action_mean": torch.as_tensor(norm.action_mean),
        "action_std": torch.as_tensor(norm.action_std),
    }


def normalizers_from_mapping(mapping: dict[str, Any]) -> Normalizers:
    return Normalizers(
        point_mean=np.asarray(mapping["point_mean"], dtype=np.float32),
        point_std=np.asarray(mapping["point_std"], dtype=np.float32),
        proprio_mean=np.asarray(mapping["proprio_mean"], dtype=np.float32),
        proprio_std=np.asarray(mapping["proprio_std"], dtype=np.float32),
        action_mean=np.asarray(mapping["action_mean"], dtype=np.float32),
        action_std=np.asarray(mapping["action_std"], dtype=np.float32),
    )


def fit_normalizers(data: LoadedData, train_idx: np.ndarray) -> Normalizers:
    train_points = data.points[train_idx]
    return Normalizers(
        point_mean=train_points.reshape(-1, train_points.shape[-1]).mean(axis=0).astype(np.float32),
        point_std=safe_std(train_points.reshape(-1, train_points.shape[-1]), axis=0),
        proprio_mean=data.proprio[train_idx].mean(axis=0).astype(np.float32),
        proprio_std=safe_std(data.proprio[train_idx], axis=0),
        action_mean=data.actions[train_idx].mean(axis=0).astype(np.float32),
        action_std=safe_std(data.actions[train_idx], axis=0),
    )


def normalize_points(points: np.ndarray, norm: Normalizers) -> np.ndarray:
    return ((points - norm.point_mean.reshape(1, 1, -1)) / norm.point_std.reshape(1, 1, -1)).astype(np.float32)


def normalize_proprio(proprio: np.ndarray, norm: Normalizers) -> np.ndarray:
    return ((proprio - norm.proprio_mean.reshape(1, -1)) / norm.proprio_std.reshape(1, -1)).astype(np.float32)


def normalize_actions(actions: np.ndarray, norm: Normalizers) -> np.ndarray:
    return ((actions - norm.action_mean.reshape(1, -1)) / norm.action_std.reshape(1, -1)).astype(np.float32)


def denormalize_actions(actions: np.ndarray, norm: Normalizers) -> np.ndarray:
    return (actions * norm.action_std.reshape(1, -1) + norm.action_mean.reshape(1, -1)).astype(np.float32)


def make_points_for_mode(
    data: LoadedData,
    indices: np.ndarray,
    mode: str,
    norm: Normalizers,
) -> np.ndarray:
    if mode == "normal":
        raw = data.points[indices]
    elif mode == "null":
        raw = np.zeros_like(data.points[indices])
    elif mode == "cross_sample":
        source = data.cross_source_indices[indices]
        raw = data.points[source]
    else:
        raise ValueError(f"unknown point mode: {mode}")
    return normalize_points(raw, norm)


def batch_indices(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: LoadedData,
    indices: np.ndarray,
    norm: Normalizers,
    mode: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    norm_losses: list[float] = []
    raw_preds: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    for idx in batch_indices(indices, batch_size):
        points = torch.as_tensor(make_points_for_mode(data, idx, mode, norm), device=device)
        proprio = torch.as_tensor(normalize_proprio(data.proprio[idx], norm), device=device)
        target = torch.as_tensor(normalize_actions(data.actions[idx], norm), device=device)
        pred = model(points, proprio)
        norm_losses.append(float(F.mse_loss(pred, target).detach().cpu()))
        raw_preds.append(denormalize_actions(pred.detach().cpu().numpy(), norm))
        raw_targets.append(data.actions[idx])

    pred_raw = np.concatenate(raw_preds, axis=0)
    target_raw = np.concatenate(raw_targets, axis=0)
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
    data: LoadedData,
    indices: np.ndarray,
    norm: Normalizers,
    other_mode: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    deltas: list[np.ndarray] = []
    for idx in batch_indices(indices, batch_size):
        normal_points = torch.as_tensor(make_points_for_mode(data, idx, "normal", norm), device=device)
        other_points = torch.as_tensor(make_points_for_mode(data, idx, other_mode, norm), device=device)
        proprio = torch.as_tensor(normalize_proprio(data.proprio[idx], norm), device=device)
        pred_normal = denormalize_actions(model(normal_points, proprio).detach().cpu().numpy(), norm)
        pred_other = denormalize_actions(model(other_points, proprio).detach().cpu().numpy(), norm)
        deltas.append(pred_normal - pred_other)
    delta = np.concatenate(deltas, axis=0)
    return {
        f"paired_normal_vs_{other_mode}_l1": float(np.mean(np.abs(delta))),
        f"paired_normal_vs_{other_mode}_l2": float(np.mean(np.linalg.norm(delta, axis=-1))),
        f"paired_normal_vs_{other_mode}_max_l2": float(np.max(np.linalg.norm(delta, axis=-1))),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_hdf5_samples(args.input, args.num_points, args.max_samples, args.seed)
    train_idx, val_idx = split_by_episode(data.episode_ids, args.val_fraction, args.seed)
    norm = fit_normalizers(data, train_idx)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PointNetActionDecoder(
        point_dim=data.points.shape[-1],
        proprio_dim=data.proprio.shape[-1],
        action_dim=data.actions.shape[-1],
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 17)

    train_losses: list[float] = []
    model.train()
    for step in range(1, args.steps + 1):
        replace = len(train_idx) < args.batch_size
        idx = rng.choice(train_idx, size=args.batch_size, replace=replace)
        points = torch.as_tensor(make_points_for_mode(data, idx, args.train_point_mode, norm), device=device)
        proprio = torch.as_tensor(normalize_proprio(data.proprio[idx], norm), device=device)
        target = torch.as_tensor(normalize_actions(data.actions[idx], norm), device=device)
        pred = model(points, proprio)
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

        if args.log_every and step % args.log_every == 0:
            recent = np.mean(train_losses[-args.log_every :])
            print(f"step {step:04d} train_norm_mse={recent:.6f}")

    metrics = {
        mode: evaluate(model, data, val_idx, norm, mode, args.eval_batch_size, device)
        for mode in ("normal", "null", "cross_sample")
    }
    deltas = {}
    deltas.update(paired_delta(model, data, val_idx, norm, "null", args.eval_batch_size, device))
    deltas.update(paired_delta(model, data, val_idx, norm, "cross_sample", args.eval_batch_size, device))

    normal_mse = metrics["normal"]["raw_mse"]
    null_mse = metrics["null"]["raw_mse"]
    cross_mse = metrics["cross_sample"]["raw_mse"]
    cross_delta = deltas["paired_normal_vs_cross_sample_l2"]
    gate_pass = bool(
        normal_mse < null_mse
        and normal_mse < cross_mse
        and cross_delta >= args.min_paired_delta
    )

    result: dict[str, Any] = {
        "input": str(args.input),
        "num_samples": int(len(data.actions)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_episodes": int(len(np.unique(data.episode_ids))),
        "num_points": int(args.num_points),
        "action_dim": int(data.actions.shape[-1]),
        "proprio_dim": int(data.proprio.shape[-1]),
        "train_point_mode": args.train_point_mode,
        "steps": int(args.steps),
        "device": str(device),
        "metrics": metrics,
        "paired_deltas": deltas,
        "gate": {
            "passed": gate_pass,
            "normal_raw_mse_lt_null": bool(normal_mse < null_mse),
            "normal_raw_mse_lt_cross_sample": bool(normal_mse < cross_mse),
            "paired_normal_vs_cross_l2": float(cross_delta),
            "min_paired_delta": float(args.min_paired_delta),
            "interpretation": (
                "GO: normal pointcloud improves action prediction and corruptions change predictions."
                if gate_pass
                else "NO-GO: pointcloud content is not yet a defensible causal action signal."
            ),
        },
        "source_examples": data.source_steps[:5],
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
                    "action_dim": int(data.actions.shape[-1]),
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
    parser.add_argument("--output", type=Path, default=Path("experiments/logs/maniskill_pointcloud_decoder_gate.json"))
    parser.add_argument("--checkpoint_output", type=Path, default=None)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--train_point_mode", choices=["normal", "null", "cross_sample"], default="normal")
    parser.add_argument("--min_paired_delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
