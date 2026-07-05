#!/usr/bin/env python3
"""Train a phase classifier on geometry-teacher object features.

The phase-conditioned teacher-distilled policy currently receives a hand-written
controller phase. This script learns that phase label from the same object
features used by the action decoder, so rollout can test whether the phase cue
can be made learned rather than rule-supplied.
"""

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

from train_object_feature_action_decoder import normalize_features, safe_std
from train_teacher_object_feature_decoder import batch_indices, split_by_episode


@dataclass
class PhaseNormalizers:
    feature_mean: np.ndarray
    feature_std: np.ndarray


class ObjectFeaturePhaseClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_phases: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_phases),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def normalizers_to_tensors(norm: PhaseNormalizers) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value) for key, value in norm.__dict__.items()}


def fit_normalizers(features: np.ndarray, train_idx: np.ndarray) -> PhaseNormalizers:
    return PhaseNormalizers(
        feature_mean=features[train_idx].mean(axis=0).astype(np.float32),
        feature_std=safe_std(features[train_idx], axis=0),
    )


def class_weights(labels: np.ndarray, train_idx: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels[train_idx], minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (float(nonzero.sum()) * counts[nonzero])
    return weights


@torch.no_grad()
def evaluate(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    norm: PhaseNormalizers,
    phase_names: list[str],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    num_phases = len(phase_names)
    confusion = np.zeros((num_phases, num_phases), dtype=np.int64)
    losses: list[float] = []
    confidences: list[float] = []
    for idx in batch_indices(indices, batch_size):
        feature_t = torch.as_tensor(normalize_features(features[idx], norm), device=device)
        label_t = torch.as_tensor(labels[idx], dtype=torch.long, device=device)
        logits = model(feature_t)
        losses.append(float(F.cross_entropy(logits, label_t).detach().cpu()))
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1).detach().cpu().numpy()
        confidences.extend(probs.max(dim=-1).values.detach().cpu().numpy().astype(float).tolist())
        for target_i, pred_i in zip(labels[idx], pred):
            confusion[int(target_i), int(pred_i)] += 1

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    per_class: dict[str, dict[str, float | int]] = {}
    class_accs: list[float] = []
    for phase_idx, name in enumerate(phase_names):
        support = int(confusion[phase_idx].sum())
        hits = int(confusion[phase_idx, phase_idx])
        acc = float(hits / support) if support else 0.0
        if support:
            class_accs.append(acc)
        per_class[name] = {"support": support, "correct": hits, "accuracy": acc}
    return {
        "cross_entropy": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(correct / total) if total else 0.0,
        "macro_accuracy": float(np.mean(class_accs)) if class_accs else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "confusion": confusion.tolist(),
        "per_class": per_class,
    }


def load_phase_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    data = np.load(path)
    features_with_phase = np.asarray(data["features"], dtype=np.float32)
    episode_ids = np.asarray(data["episode_ids"], dtype=np.int64)
    include_phase = bool(int(np.asarray(data.get("include_phase", [0])).reshape(-1)[0]))
    phase_names = [str(item) for item in np.asarray(data.get("phase_names", []), dtype=str).reshape(-1).tolist()]
    if not include_phase or not phase_names:
        raise ValueError(f"{path} must contain features with appended phase one-hot labels")
    num_phases = len(phase_names)
    if features_with_phase.shape[-1] <= num_phases:
        raise ValueError("feature dimension is too small for appended phase labels")
    phase_one_hot = features_with_phase[:, -num_phases:]
    labels = phase_one_hot.argmax(axis=-1).astype(np.int64)
    if not np.allclose(phase_one_hot.sum(axis=-1), 1.0):
        raise ValueError("phase labels are expected to be one-hot in the last columns")
    features = features_with_phase[:, :-num_phases].astype(np.float32)
    cube_seg_id = int(np.asarray(data.get("cube_seg_id", [18])).reshape(-1)[0])
    return features, labels, episode_ids, phase_names, cube_seg_id


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    features, labels, episode_ids, phase_names, cube_seg_id = load_phase_dataset(args.input)
    train_idx, val_idx = split_by_episode(episode_ids, args.val_fraction, args.seed)
    norm = fit_normalizers(features, train_idx)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ObjectFeaturePhaseClassifier(features.shape[-1], len(phase_names), args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights_np = class_weights(labels, train_idx, len(phase_names)) if args.class_weighted else np.ones(len(phase_names), dtype=np.float32)
    weights = torch.as_tensor(weights_np, device=device)
    rng = np.random.default_rng(args.seed + 29)

    for step in range(1, args.steps + 1):
        idx = rng.choice(train_idx, size=args.batch_size, replace=len(train_idx) < args.batch_size)
        feature_t = torch.as_tensor(normalize_features(features[idx], norm), device=device)
        label_t = torch.as_tensor(labels[idx], dtype=torch.long, device=device)
        logits = model(feature_t)
        loss = F.cross_entropy(logits, label_t, weight=weights, label_smoothing=args.label_smoothing)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if args.log_every and step % args.log_every == 0:
            print(f"step {step:04d} train_ce={float(loss.detach().cpu()):.6f}")

    metrics = {
        "train": evaluate(model, features, labels, train_idx, norm, phase_names, args.eval_batch_size, device),
        "val": evaluate(model, features, labels, val_idx, norm, phase_names, args.eval_batch_size, device),
    }
    counts = np.bincount(labels, minlength=len(phase_names)).astype(int)
    result: dict[str, Any] = {
        "input": str(args.input),
        "num_samples": int(len(labels)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_episodes": int(len(np.unique(episode_ids))),
        "feature_dim": int(features.shape[-1]),
        "num_phases": int(len(phase_names)),
        "phase_names": phase_names,
        "phase_counts": {name: int(counts[idx]) for idx, name in enumerate(phase_names)},
        "class_weighted": bool(args.class_weighted),
        "class_weights": {name: float(weights_np[idx]) for idx, name in enumerate(phase_names)},
        "steps": int(args.steps),
        "metrics": metrics,
        "note": "Single-frame object-feature classifier for replacing hand-written phase in PickCube rollouts.",
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
                    "num_phases": int(len(phase_names)),
                    "hidden_dim": int(args.hidden_dim),
                    "phase_names": phase_names,
                    "cube_seg_id": int(cube_seg_id),
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
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--class_weighted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=1000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
