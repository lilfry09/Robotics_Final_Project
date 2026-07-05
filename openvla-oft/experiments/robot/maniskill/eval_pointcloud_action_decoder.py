#!/usr/bin/env python3
"""Closed-loop smoke evaluation for the tiny ManiSkill point-cloud decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_pointcloud_action_decoder import (
    PointNetActionDecoder,
    denormalize_actions,
    load_hdf5_samples,
    normalizers_from_mapping,
    normalize_points,
    normalize_proprio,
    select_points,
)


def require_maniskill() -> Any:
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment diagnostic
        raise SystemExit(
            "ManiSkill3 is not available. Use the isolated venv at "
            "`/root/autodl-tmp/envs/maniskill3-venv/bin/python`. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return gym


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    else:
        value = np.asarray(value)
    return np.asarray(value)


def squeeze_batch(value: Any) -> np.ndarray:
    arr = to_numpy(value)
    if arr.shape[:1] == (1,):
        arr = arr[0]
    return arr


def scalar_bool(value: Any) -> bool:
    arr = to_numpy(value)
    return bool(arr.reshape(-1)[0])


def load_checkpoint(path: Path, device: torch.device) -> tuple[PointNetActionDecoder, Any, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = PointNetActionDecoder(
        point_dim=int(config["point_dim"]),
        proprio_dim=int(config["proprio_dim"]),
        action_dim=int(config["action_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    norm_mapping = {
        key: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        for key, value in checkpoint["normalizers"].items()
    }
    norm = normalizers_from_mapping(norm_mapping)
    return model, norm, config


def extract_proprio(obs: dict[str, Any]) -> np.ndarray:
    agent = obs["agent"]
    qpos = squeeze_batch(agent["qpos"]).reshape(-1).astype(np.float32)
    qvel = squeeze_batch(agent["qvel"]).reshape(-1).astype(np.float32)
    return np.concatenate([qpos, qvel], axis=0).astype(np.float32)


def extract_points(obs: dict[str, Any], num_points: int, rng: np.random.Generator) -> np.ndarray:
    pointcloud = obs["pointcloud"]
    xyzw = squeeze_batch(pointcloud["xyzw"])
    rgb = squeeze_batch(pointcloud["rgb"]) if "rgb" in pointcloud else None
    return select_points(xyzw, rgb, num_points, rng)


def choose_points(
    obs: dict[str, Any],
    point_mode: str,
    num_points: int,
    rng: np.random.Generator,
    cross_pool: np.ndarray | None,
) -> np.ndarray:
    if point_mode == "normal":
        return extract_points(obs, num_points, rng)
    if point_mode == "null":
        return np.zeros((num_points, 6), dtype=np.float32)
    if point_mode == "cross_demo":
        if cross_pool is None or len(cross_pool) == 0:
            raise ValueError("cross_demo requires --cross_hdf5")
        return cross_pool[int(rng.integers(0, len(cross_pool)))]
    raise ValueError(f"unknown point mode: {point_mode}")


@torch.no_grad()
def predict_action(
    model: PointNetActionDecoder,
    norm: Any,
    points: np.ndarray,
    proprio: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    norm_points = normalize_points(points[None], norm)
    norm_proprio = normalize_proprio(proprio[None], norm)
    pred_norm = model(
        torch.as_tensor(norm_points, device=device),
        torch.as_tensor(norm_proprio, device=device),
    )
    pred = denormalize_actions(pred_norm.detach().cpu().numpy(), norm)[0]
    return pred.astype(np.float32)


def clip_action(action: np.ndarray, action_space: Any) -> np.ndarray:
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    return np.clip(action, low, high).astype(np.float32)


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

    for episode_idx in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode_idx)
        rewards: list[float] = []
        action_norms: list[float] = []
        successes: list[bool] = []
        terminated = False
        truncated = False
        final_info: dict[str, Any] = {}

        for step_idx in range(args.max_steps):
            points = choose_points(obs, args.point_mode, num_points, rng, cross_pool)
            proprio = extract_proprio(obs)
            action = predict_action(model, norm, points, proprio, device)
            action = clip_action(action, env.action_space)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(to_numpy(reward).reshape(-1)[0]))
            action_norms.append(float(np.linalg.norm(action)))
            if isinstance(info, dict) and "success" in info:
                successes.append(scalar_bool(info["success"]))
            final_info = info if isinstance(info, dict) else {}
            if scalar_bool(terminated) or scalar_bool(truncated):
                break

        success = bool(successes[-1]) if successes else bool(final_info.get("success", False))
        episodes.append(
            {
                "episode": episode_idx,
                "success": success,
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
        "note": "Closed-loop smoke for tiny offline decoder; not an OpenVLA RGB-D result.",
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
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
