#!/usr/bin/env python3
"""Closed-loop smoke for the goal-conditioned point-cloud action decoder."""

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
from train_pointcloud_action_decoder import normalize_points
from train_pointcloud_goal_action_decoder import (
    GoalNormalizers,
    PointNetGoalActionDecoder,
    denormalize_actions,
    normalize_goal,
    normalize_proprio,
)


def load_checkpoint(path: Path, device: torch.device) -> tuple[PointNetGoalActionDecoder, GoalNormalizers, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = PointNetGoalActionDecoder(
        point_dim=int(config["point_dim"]),
        proprio_dim=int(config["proprio_dim"]),
        goal_dim=int(config["goal_dim"]),
        action_dim=int(config["action_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    norm_mapping = {
        key: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        for key, value in checkpoint["normalizers"].items()
    }
    norm = GoalNormalizers(**{key: np.asarray(value, dtype=np.float32) for key, value in norm_mapping.items()})
    return model, norm, config


def extract_goal(obs: dict[str, Any]) -> np.ndarray:
    value = obs["extra"]["goal_pos"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    arr = np.asarray(value)
    if arr.shape[:1] == (1,):
        arr = arr[0]
    return arr.reshape(-1)[:3].astype(np.float32)


@torch.no_grad()
def predict_action(
    model: PointNetGoalActionDecoder,
    norm: GoalNormalizers,
    points: np.ndarray,
    proprio: np.ndarray,
    goal: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    pred_norm = model(
        torch.as_tensor(normalize_points(points[None], norm), device=device),
        torch.as_tensor(normalize_proprio(proprio[None], norm), device=device),
        torch.as_tensor(normalize_goal(goal[None], norm), device=device),
    )
    return denormalize_actions(pred_norm.detach().cpu().numpy(), norm)[0].astype(np.float32)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gym = require_maniskill()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, norm, config = load_checkpoint(args.checkpoint, device)
    num_points = int(config["num_points"])
    cross_pool = None
    if args.cross_hdf5 is not None:
        cross_data = load_hdf5_samples(args.cross_hdf5, num_points, args.cross_samples, args.seed + 313)
        cross_pool = cross_data.points

    env = gym.make(args.env_id, obs_mode="pointcloud", control_mode=args.control_mode, max_episode_steps=args.max_steps)
    rng = np.random.default_rng(args.seed)
    episodes: list[dict[str, Any]] = []
    for episode_idx in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode_idx)
        rewards: list[float] = []
        successes: list[bool] = []
        action_norms: list[float] = []
        terminated = False
        truncated = False
        for _ in range(args.max_steps):
            points = choose_points(obs, args.point_mode, num_points, rng, cross_pool)
            proprio = extract_proprio(obs)
            goal = extract_goal(obs)
            action = predict_action(model, norm, points, proprio, goal, device)
            action = clip_action(action, env.action_space)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(to_numpy(reward).reshape(-1)[0]))
            action_norms.append(float(np.linalg.norm(action)))
            if isinstance(info, dict) and "success" in info:
                successes.append(scalar_bool(info["success"]))
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
        "episodes": episodes,
        "success_rate": float(np.mean([ep["success"] for ep in episodes])) if episodes else 0.0,
        "mean_length": float(np.mean([ep["length"] for ep in episodes])) if episodes else 0.0,
        "mean_reward_sum": float(np.mean([ep["reward_sum"] for ep in episodes])) if episodes else 0.0,
        "device": str(device),
        "note": "Goal-conditioned pointcloud decoder smoke; not an OpenVLA RGB-D result.",
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
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
