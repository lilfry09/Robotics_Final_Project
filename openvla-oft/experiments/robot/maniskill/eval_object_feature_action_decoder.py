#!/usr/bin/env python3
"""Closed-loop eval for the object-centric ManiSkill action decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval_pointcloud_action_decoder import clip_action, extract_proprio, require_maniskill, scalar_bool, squeeze_batch, to_numpy
from eval_pointcloud_geometry_controller import ControllerState, cube_center_from_obs, load_cross_cube_centers, step_controller
from train_object_feature_action_decoder import (
    FeatureNormalizers,
    ObjectFeatureActionDecoder,
    denormalize_actions,
    make_feature,
    normalize_features,
)
from train_object_feature_phase_classifier import ObjectFeaturePhaseClassifier, PhaseNormalizers


DEFAULT_PHASE_NAMES = ("approach", "descend", "close", "lift", "move_goal", "hold_goal", "no_cube")


def phase_one_hot(phase: str, phase_names: list[str]) -> np.ndarray:
    values = np.zeros(len(phase_names), dtype=np.float32)
    if phase in phase_names:
        values[phase_names.index(phase)] = 1.0
    return values


def load_checkpoint(path: Path, device: torch.device) -> tuple[ObjectFeatureActionDecoder, FeatureNormalizers, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = ObjectFeatureActionDecoder(
        feature_dim=int(config["feature_dim"]),
        action_dim=int(config["action_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    norm_mapping = {
        key: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        for key, value in checkpoint["normalizers"].items()
    }
    norm = FeatureNormalizers(**{key: np.asarray(value, dtype=np.float32) for key, value in norm_mapping.items()})
    return model, norm, config


def load_phase_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[ObjectFeaturePhaseClassifier, PhaseNormalizers, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = ObjectFeaturePhaseClassifier(
        feature_dim=int(config["feature_dim"]),
        num_phases=int(config["num_phases"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    norm_mapping = {
        key: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        for key, value in checkpoint["normalizers"].items()
    }
    norm = PhaseNormalizers(**{key: np.asarray(value, dtype=np.float32) for key, value in norm_mapping.items()})
    return model, norm, config


def extract_goal(obs: dict[str, Any]) -> np.ndarray:
    return squeeze_batch(obs["extra"]["goal_pos"]).reshape(-1)[:3].astype(np.float32)


def extract_tcp(obs: dict[str, Any]) -> np.ndarray:
    return squeeze_batch(obs["extra"]["tcp_pose"]).reshape(-1)[:3].astype(np.float32)


def extract_grasped(obs: dict[str, Any]) -> float:
    return float(scalar_bool(obs["extra"]["is_grasped"]))


def choose_cube_center(
    obs: dict[str, Any],
    point_mode: str,
    cube_seg_id: int,
    cross_centers: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    if point_mode == "normal":
        center = cube_center_from_obs(obs, cube_seg_id)
        if center is None:
            return np.zeros(3, dtype=np.float32), 0.0
        return center.astype(np.float32), 1.0
    if point_mode == "null":
        return np.zeros(3, dtype=np.float32), 0.0
    if point_mode == "cross_demo":
        if cross_centers is None:
            raise ValueError("cross_demo requires --cross_hdf5")
        return cross_centers[int(rng.integers(0, len(cross_centers)))].astype(np.float32), 1.0
    if point_mode == "same":
        raise ValueError("`same` must be resolved before choose_cube_center")
    raise ValueError(f"unknown point mode: {point_mode}")


def resolve_phase_source(phase_source: str, point_mode: str) -> str:
    return point_mode if phase_source == "same" else phase_source


@torch.no_grad()
def predict_action(
    model: ObjectFeatureActionDecoder,
    norm: FeatureNormalizers,
    feature: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    pred_norm = model(torch.as_tensor(normalize_features(feature[None], norm), device=device))
    return denormalize_actions(pred_norm.detach().cpu().numpy(), norm)[0].astype(np.float32)


@torch.no_grad()
def predict_phase(
    model: ObjectFeaturePhaseClassifier,
    norm: PhaseNormalizers,
    feature: np.ndarray,
    device: torch.device,
) -> tuple[int, float, np.ndarray]:
    logits = model(torch.as_tensor(normalize_features(feature[None], norm), device=device))
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0].astype(np.float32)
    phase_idx = int(np.argmax(probs))
    return phase_idx, float(probs[phase_idx]), probs


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gym = require_maniskill()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, norm, config = load_checkpoint(args.checkpoint, device)
    phase_model = None
    phase_norm = None
    phase_config: dict[str, Any] = {}
    if args.phase_mode == "predicted":
        if args.phase_model_checkpoint is None:
            raise ValueError("--phase_mode predicted requires --phase_model_checkpoint")
        phase_model, phase_norm, phase_config = load_phase_checkpoint(args.phase_model_checkpoint, device)
    cube_seg_id = int(config.get("cube_seg_id", args.cube_seg_id))
    cross_centers = None
    if args.cross_hdf5 is not None:
        cross_centers = load_cross_cube_centers(args.cross_hdf5, cube_seg_id, args.cross_samples, args.seed + 97)
    include_phase = bool(config.get("include_phase", False))
    phase_names = [str(item) for item in config.get("phase_names", DEFAULT_PHASE_NAMES)]
    phase_source = resolve_phase_source(args.phase_source, args.point_mode)
    predicted_phase_source = resolve_phase_source(args.predicted_phase_source, args.point_mode)
    if phase_source == "cross_demo" and cross_centers is None:
        raise ValueError("phase_source=cross_demo requires --cross_hdf5")
    if predicted_phase_source == "cross_demo" and cross_centers is None:
        raise ValueError("predicted_phase_source=cross_demo requires --cross_hdf5")
    if args.phase_mode == "predicted" and not include_phase:
        raise ValueError("--phase_mode predicted only applies to checkpoints trained with include_phase")
    if args.phase_mode == "predicted" and phase_config:
        classifier_names = [str(item) for item in phase_config.get("phase_names", [])]
        if classifier_names and classifier_names != phase_names:
            raise ValueError(f"phase name mismatch: action={phase_names}, classifier={classifier_names}")

    env = gym.make(args.env_id, obs_mode="pointcloud", control_mode=args.control_mode, max_episode_steps=args.max_steps)
    rng = np.random.default_rng(args.seed)
    episodes: list[dict[str, Any]] = []
    for episode_idx in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode_idx)
        phase_state = ControllerState()
        rewards: list[float] = []
        successes: list[bool] = []
        action_norms: list[float] = []
        phase_records: list[str] = []
        phase_confidences: list[float] = []
        terminated = False
        truncated = False
        final_cube_valid = 0.0
        final_debug: dict[str, Any] = {}
        for _ in range(args.max_steps):
            phase_before = phase_state.phase
            cube_center, cube_valid = choose_cube_center(obs, args.point_mode, cube_seg_id, cross_centers, rng)
            final_cube_valid = cube_valid
            tcp = extract_tcp(obs)
            goal = extract_goal(obs)
            grasped = extract_grasped(obs)
            feature = make_feature(
                cube_center,
                cube_valid,
                tcp,
                goal,
                grasped,
                extract_proprio(obs),
            )
            if include_phase:
                if args.phase_mode == "state_machine":
                    phase_vector = phase_one_hot(phase_before, phase_names)
                    phase_records.append(phase_before)
                    phase_center, phase_valid = choose_cube_center(obs, phase_source, cube_seg_id, cross_centers, rng)
                    phase_cube = phase_center if phase_valid > 0.5 else None
                    step_controller(obs, phase_state, phase_cube, args.phase_gain, args.phase_max_xyz_action)
                elif args.phase_mode == "predicted":
                    assert phase_model is not None and phase_norm is not None
                    phase_center, phase_valid = choose_cube_center(
                        obs, predicted_phase_source, cube_seg_id, cross_centers, rng
                    )
                    phase_feature = make_feature(
                        phase_center,
                        phase_valid,
                        tcp,
                        goal,
                        grasped,
                        extract_proprio(obs),
                    )
                    phase_idx, phase_confidence, phase_probs = predict_phase(
                        phase_model,
                        phase_norm,
                        phase_feature,
                        device,
                    )
                    if args.predicted_phase_format == "soft":
                        phase_vector = phase_probs
                    else:
                        phase_vector = np.zeros(len(phase_names), dtype=np.float32)
                        phase_vector[phase_idx] = 1.0
                    phase_records.append(phase_names[phase_idx])
                    phase_confidences.append(phase_confidence)
                else:
                    raise ValueError(f"unknown phase_mode: {args.phase_mode}")
                feature = np.concatenate([feature, phase_vector], axis=0).astype(np.float32)
            action = predict_action(model, norm, feature, device)
            action = clip_action(action, env.action_space)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(to_numpy(reward).reshape(-1)[0]))
            action_norms.append(float(np.linalg.norm(action)))
            if isinstance(info, dict) and "success" in info:
                successes.append(scalar_bool(info["success"]))
            if scalar_bool(terminated) or scalar_bool(truncated):
                break
        final_cube, final_valid = choose_cube_center(obs, args.point_mode, cube_seg_id, cross_centers, rng)
        final_tcp = extract_tcp(obs)
        final_goal = extract_goal(obs)
        final_debug = {
            "cube": final_cube.tolist(),
            "cube_valid": float(final_valid),
            "tcp": final_tcp.tolist(),
            "goal": final_goal.tolist(),
            "grasped": bool(extract_grasped(obs)),
            "tcp_to_goal_norm": float(np.linalg.norm(final_tcp - final_goal)),
            "cube_to_goal_norm": float(np.linalg.norm(final_cube - final_goal)) if final_valid > 0.5 else None,
            "tcp_to_cube_norm": float(np.linalg.norm(final_tcp - final_cube)) if final_valid > 0.5 else None,
        }
        episodes.append(
            {
                "episode": episode_idx,
                "success": bool(successes[-1]) if successes else False,
                "length": len(rewards),
                "reward_sum": float(np.sum(rewards)),
                "action_norm_mean": float(np.mean(action_norms)) if action_norms else 0.0,
                "final_cube_valid": float(final_cube_valid),
                "final_debug": final_debug,
                "phase_counts": {phase: phase_records.count(phase) for phase in sorted(set(phase_records))},
                "phase_confidence_mean": float(np.mean(phase_confidences)) if phase_confidences else None,
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
        "include_phase": include_phase,
        "phase_mode": args.phase_mode,
        "phase_source": phase_source,
        "predicted_phase_source": predicted_phase_source,
        "predicted_phase_format": args.predicted_phase_format,
        "phase_model_checkpoint": str(args.phase_model_checkpoint) if args.phase_model_checkpoint else None,
        "note": "Object-centric learned BC using pointcloud segmentation features; not an OpenVLA result.",
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
    parser.add_argument("--phase_mode", choices=["state_machine", "predicted"], default="state_machine")
    parser.add_argument("--phase_model_checkpoint", type=Path, default=None)
    parser.add_argument("--phase_source", choices=["same", "normal", "null", "cross_demo"], default="same")
    parser.add_argument("--predicted_phase_source", choices=["same", "normal", "null", "cross_demo"], default="same")
    parser.add_argument("--predicted_phase_format", choices=["hard", "soft"], default="hard")
    parser.add_argument("--cross_hdf5", type=Path, default=None)
    parser.add_argument("--cross_samples", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--cube_seg_id", type=int, default=18)
    parser.add_argument("--phase_gain", type=float, default=5.0)
    parser.add_argument("--phase_max_xyz_action", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
