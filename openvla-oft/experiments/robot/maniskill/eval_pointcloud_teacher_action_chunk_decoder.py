#!/usr/bin/env python3
"""Closed-loop eval for the raw-pointcloud teacher action-chunk decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from collect_geometry_teacher_pointcloud_dataset import extract_points, make_task_feature
from eval_pointcloud_action_decoder import clip_action, require_maniskill, scalar_bool, to_numpy
from train_pointcloud_teacher_action_chunk_decoder import (
    ChunkNormalizers,
    PointNetTeacherActionChunkDecoder,
    denormalize_chunks,
    normalize_points,
    normalize_task,
)


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[PointNetTeacherActionChunkDecoder, ChunkNormalizers, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = PointNetTeacherActionChunkDecoder(
        point_dim=int(config["point_dim"]),
        task_dim=int(config["task_dim"]),
        action_dim=int(config["action_dim"]),
        num_phases=int(config["num_phases"]),
        chunk_horizon=int(config["chunk_horizon"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    norm_mapping = {
        key: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        for key, value in checkpoint["normalizers"].items()
    }
    norm = ChunkNormalizers(**{key: np.asarray(value, dtype=np.float32) for key, value in norm_mapping.items()})
    return model, norm, config


def load_cross_points(path: Path) -> np.ndarray:
    data = np.load(path)
    return np.asarray(data["points"], dtype=np.float32)


def choose_points(
    obs: dict[str, Any],
    point_mode: str,
    num_points: int,
    rng: np.random.Generator,
    cross_pool: np.ndarray | None,
    min_z: float,
) -> np.ndarray:
    if point_mode == "normal":
        return extract_points(obs, num_points, rng, min_z)
    if point_mode == "null":
        return np.zeros((num_points, 6), dtype=np.float32)
    if point_mode == "cross_demo":
        if cross_pool is None:
            raise ValueError("cross_demo requires --cross_dataset")
        return cross_pool[int(rng.integers(0, len(cross_pool)))]
    raise ValueError(f"unknown point mode: {point_mode}")


@torch.no_grad()
def predict_chunk(
    model: PointNetTeacherActionChunkDecoder,
    norm: ChunkNormalizers,
    points: np.ndarray,
    task_feature: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, int, float]:
    pred_norm, logits, _ = model(
        torch.as_tensor(normalize_points(points[None], norm), device=device),
        torch.as_tensor(normalize_task(task_feature[None], norm), device=device),
        None,
    )
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0].astype(np.float32)
    phase_idx = int(np.argmax(probs))
    chunk = denormalize_chunks(pred_norm.detach().cpu().numpy(), norm)[0].astype(np.float32)
    return chunk, phase_idx, float(probs[phase_idx])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gym = require_maniskill()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, norm, config = load_checkpoint(args.checkpoint, device)
    num_points = int(config["num_points"])
    chunk_horizon = int(config["chunk_horizon"])
    phase_names = [str(item) for item in config.get("phase_names", [])]
    execute_steps = max(1, min(args.execute_steps, chunk_horizon))
    cross_pool = load_cross_points(args.cross_dataset) if args.cross_dataset is not None else None
    env = gym.make(args.env_id, obs_mode="pointcloud", control_mode=args.control_mode, max_episode_steps=args.max_steps)
    rng = np.random.default_rng(args.seed)
    episodes: list[dict[str, Any]] = []
    for episode_idx in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode_idx)
        rewards: list[float] = []
        successes: list[bool] = []
        action_norms: list[float] = []
        phases: list[str] = []
        confidences: list[float] = []
        terminated = False
        truncated = False
        step_idx = 0
        while step_idx < args.max_steps:
            points = choose_points(obs, args.point_mode, num_points, rng, cross_pool, args.min_z)
            task_feature = make_task_feature(obs)
            chunk, phase_idx, phase_conf = predict_chunk(model, norm, points, task_feature, device)
            for action in chunk[:execute_steps]:
                action = clip_action(action, env.action_space)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(to_numpy(reward).reshape(-1)[0]))
                action_norms.append(float(np.linalg.norm(action)))
                confidences.append(phase_conf)
                if phase_names:
                    phases.append(phase_names[phase_idx])
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
                "phase_confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
                "phase_counts": {phase: phases.count(phase) for phase in sorted(set(phases))},
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
        "chunk_horizon": chunk_horizon,
        "episodes": episodes,
        "success_rate": float(np.mean([ep["success"] for ep in episodes])) if episodes else 0.0,
        "mean_length": float(np.mean([ep["length"] for ep in episodes])) if episodes else 0.0,
        "mean_reward_sum": float(np.mean([ep["reward_sum"] for ep in episodes])) if episodes else 0.0,
        "min_z": float(args.min_z),
        "device": str(device),
        "note": "Raw pointcloud/task teacher-distilled action-chunk decoder; not OpenVLA.",
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
    parser.add_argument("--cross_dataset", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--execute_steps", type=int, default=2)
    parser.add_argument("--min_z", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
