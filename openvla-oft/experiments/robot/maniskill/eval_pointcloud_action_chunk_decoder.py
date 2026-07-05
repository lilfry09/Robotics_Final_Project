#!/usr/bin/env python3
"""Closed-loop smoke evaluation for a ManiSkill point-cloud action chunk decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval_pointcloud_action_decoder import (
    choose_points,
    clip_action,
    extract_proprio,
    load_hdf5_samples,
    require_maniskill,
    scalar_bool,
    to_numpy,
)
from train_pointcloud_action_chunk_decoder import PointNetActionChunkDecoder, denormalize_chunks
from train_pointcloud_action_decoder import normalizers_from_mapping, normalize_points, normalize_proprio


def load_checkpoint(path: Path, device: torch.device) -> tuple[PointNetActionChunkDecoder, Any, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = PointNetActionChunkDecoder(
        point_dim=int(config["point_dim"]),
        proprio_dim=int(config["proprio_dim"]),
        action_dim=int(config["action_dim"]),
        chunk_horizon=int(config["chunk_horizon"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    norm_mapping = {
        key: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        for key, value in checkpoint["normalizers"].items()
    }
    return model, normalizers_from_mapping(norm_mapping), config


@torch.no_grad()
def predict_chunk(
    model: PointNetActionChunkDecoder,
    norm: Any,
    points: np.ndarray,
    proprio: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    pred_norm = model(
        torch.as_tensor(normalize_points(points[None], norm), device=device),
        torch.as_tensor(normalize_proprio(proprio[None], norm), device=device),
    )
    return denormalize_chunks(pred_norm.detach().cpu().numpy(), norm)[0].astype(np.float32)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gym = require_maniskill()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, norm, config = load_checkpoint(args.checkpoint, device)
    num_points = int(config["num_points"])

    cross_pool = None
    if args.cross_hdf5 is not None:
        cross_data = load_hdf5_samples(args.cross_hdf5, num_points, args.cross_samples, args.seed + 313)
        cross_pool = cross_data.points

    env = gym.make(
        args.env_id,
        obs_mode="pointcloud",
        control_mode=args.control_mode,
        max_episode_steps=args.max_steps,
    )
    rng = np.random.default_rng(args.seed)
    episodes: list[dict[str, Any]] = []
    execute_steps = min(args.execute_steps, int(config["chunk_horizon"]))

    for episode_idx in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode_idx)
        rewards: list[float] = []
        action_norms: list[float] = []
        successes: list[bool] = []
        terminated = False
        truncated = False
        step_idx = 0
        while step_idx < args.max_steps:
            points = choose_points(obs, args.point_mode, num_points, rng, cross_pool)
            proprio = extract_proprio(obs)
            chunk = predict_chunk(model, norm, points, proprio, device)
            for action in chunk[:execute_steps]:
                action = clip_action(action, env.action_space)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(to_numpy(reward).reshape(-1)[0]))
                action_norms.append(float(np.linalg.norm(action)))
                if isinstance(info, dict) and "success" in info:
                    successes.append(scalar_bool(info["success"]))
                step_idx += 1
                if scalar_bool(terminated) or scalar_bool(truncated) or step_idx >= args.max_steps:
                    break
            if scalar_bool(terminated) or scalar_bool(truncated):
                break
        episodes.append(
            {
                "episode": episode_idx,
                "success": bool(successes[-1]) if successes else False,
                "length": len(rewards),
                "reward_sum": float(np.sum(rewards)),
                "action_norm_mean": float(np.mean(action_norms)) if action_norms else 0.0,
                "terminated": scalar_bool(terminated),
                "truncated": scalar_bool(truncated),
            }
        )

    env.close()
    result = {
        "checkpoint": str(args.checkpoint),
        "env_id": args.env_id,
        "control_mode": args.control_mode,
        "point_mode": args.point_mode,
        "execute_steps": execute_steps,
        "episodes": episodes,
        "success_rate": float(np.mean([ep["success"] for ep in episodes])) if episodes else 0.0,
        "mean_length": float(np.mean([ep["length"] for ep in episodes])) if episodes else 0.0,
        "mean_reward_sum": float(np.mean([ep["reward_sum"] for ep in episodes])) if episodes else 0.0,
        "device": str(device),
        "note": "Closed-loop smoke for tiny offline chunk decoder; not an OpenVLA RGB-D result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("success_rate", "mean_length", "mean_reward_sum")}, indent=2))
    print(f"wrote {args.output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env_id", default="PickCube-v1")
    parser.add_argument("--control_mode", default="pd_ee_delta_pos")
    parser.add_argument("--point_mode", choices=["normal", "null", "cross_demo"], default="normal")
    parser.add_argument("--cross_hdf5", type=Path, default=None)
    parser.add_argument("--cross_samples", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--execute_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
