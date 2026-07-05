#!/usr/bin/env python3
"""Train an object-feature policy on geometry-controller teacher rollouts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_object_feature_action_decoder import (
    FeatureNormalizers,
    ObjectFeatureActionDecoder,
    denormalize_actions,
    normalize_actions,
    normalize_features,
    normalizers_to_tensors,
)
from train_pointcloud_action_decoder import safe_std


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


def fit_normalizers(features: np.ndarray, actions: np.ndarray, train_idx: np.ndarray) -> FeatureNormalizers:
    return FeatureNormalizers(
        feature_mean=features[train_idx].mean(axis=0).astype(np.float32),
        feature_std=safe_std(features[train_idx], axis=0),
        action_mean=actions[train_idx].mean(axis=0).astype(np.float32),
        action_std=safe_std(actions[train_idx], axis=0),
    )


def batch_indices(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    features: np.ndarray,
    actions: np.ndarray,
    indices: np.ndarray,
    norm: FeatureNormalizers,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    norm_losses: list[float] = []
    for idx in batch_indices(indices, batch_size):
        feature_t = torch.as_tensor(normalize_features(features[idx], norm), device=device)
        target_t = torch.as_tensor(normalize_actions(actions[idx], norm), device=device)
        pred = model(feature_t)
        norm_losses.append(float(F.mse_loss(pred, target_t).detach().cpu()))
        preds.append(denormalize_actions(pred.detach().cpu().numpy(), norm))
        targets.append(actions[idx])
    pred_raw = np.concatenate(preds, axis=0)
    target_raw = np.concatenate(targets, axis=0)
    err = pred_raw - target_raw
    return {
        "norm_mse": float(np.mean(norm_losses)),
        "raw_mse": float(np.mean(err**2)),
        "raw_rmse": float(np.sqrt(np.mean(err**2))),
        "raw_l1": float(np.mean(np.abs(err))),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.input)
    features = np.asarray(data["features"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    episode_ids = np.asarray(data["episode_ids"], dtype=np.int64)
    cube_seg_id = int(np.asarray(data.get("cube_seg_id", [args.cube_seg_id])).reshape(-1)[0])
    include_phase = bool(int(np.asarray(data.get("include_phase", [0])).reshape(-1)[0]))
    phase_names = [str(item) for item in np.asarray(data.get("phase_names", []), dtype=str).reshape(-1).tolist()]
    train_idx, val_idx = split_by_episode(episode_ids, args.val_fraction, args.seed)
    norm = fit_normalizers(features, actions, train_idx)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ObjectFeatureActionDecoder(features.shape[-1], actions.shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 17)
    for step in range(1, args.steps + 1):
        idx = rng.choice(train_idx, size=args.batch_size, replace=len(train_idx) < args.batch_size)
        feature_t = torch.as_tensor(normalize_features(features[idx], norm), device=device)
        target_t = torch.as_tensor(normalize_actions(actions[idx], norm), device=device)
        pred = model(feature_t)
        loss = F.mse_loss(pred, target_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if args.log_every and step % args.log_every == 0:
            print(f"step {step:04d} train_norm_mse={float(loss.detach().cpu()):.6f}")

    metrics = {
        "train": evaluate(model, features, actions, train_idx, norm, args.eval_batch_size, device),
        "val": evaluate(model, features, actions, val_idx, norm, args.eval_batch_size, device),
    }
    result: dict[str, Any] = {
        "input": str(args.input),
        "num_samples": int(len(actions)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_episodes": int(len(np.unique(episode_ids))),
        "feature_dim": int(features.shape[-1]),
        "action_dim": int(actions.shape[-1]),
        "cube_seg_id": cube_seg_id,
        "include_phase": include_phase,
        "phase_names": phase_names,
        "steps": int(args.steps),
        "metrics": metrics,
        "note": "Teacher-distilled object-feature policy trained on geometry-controller rollouts.",
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
                    "feature_dim": int(features.shape[-1]),
                    "action_dim": int(actions.shape[-1]),
                    "hidden_dim": int(args.hidden_dim),
                    "cube_seg_id": cube_seg_id,
                    "include_phase": include_phase,
                    "phase_names": phase_names,
                },
                "train_config": {"input": str(args.input), "steps": int(args.steps), "seed": int(args.seed)},
                "result": result,
            },
            args.checkpoint_output,
        )
        print(f"wrote {args.checkpoint_output}")
    print(json.dumps(metrics["val"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint_output", type=Path, default=None)
    parser.add_argument("--cube_seg_id", type=int, default=18)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=1000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
