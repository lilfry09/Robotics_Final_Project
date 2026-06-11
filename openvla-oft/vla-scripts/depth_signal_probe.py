"""Offline probe for whether RGB-D HDF5 depth carries task-relevant signal.

This script intentionally does not load OpenVLA. It trains a small MLP on
coarse metric-depth geometry features and reports how much prediction quality
drops when depth is nulled or shuffled at evaluation time.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


TARGETS = ("action_xyz", "ee_delta_xyz", "contact_xyz", "contact_distance")


@dataclass
class ProbeArrays:
    normal: np.ndarray
    null: np.ndarray
    shuffle: np.ndarray
    target: np.ndarray


def list_hdf5_files(data_dir: Path) -> list[Path]:
    files = sorted(list(data_dir.glob("*.hdf5")) + list(data_dir.glob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")
    return files


def backproject_depth(depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    height, width = depth.shape
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")

    fx = max(float(intrinsics[0, 0]), 1e-6)
    fy = max(float(intrinsics[1, 1]), 1e-6)
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    valid = np.isfinite(depth) & (depth >= 0.01) & (depth <= 5.0)
    z_cam = depth
    x_cam = (xs - cx) * z_cam / fx
    y_cam = (ys - cy) * z_cam / fy
    xyz1_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)
    xyz_base = np.einsum("ij,hwj->hwi", extrinsics.astype(np.float32), xyz1_cam)[..., :3]
    return xyz_base, valid


def coarse_geometry_features(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    xyz_base, valid = backproject_depth(depth, intrinsics, extrinsics)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    height, width = depth.shape
    features: list[np.ndarray] = []
    for gy in range(grid_size):
        y0 = round(gy * height / grid_size)
        y1 = round((gy + 1) * height / grid_size)
        for gx in range(grid_size):
            x0 = round(gx * width / grid_size)
            x1 = round((gx + 1) * width / grid_size)
            cell_valid = valid[y0:y1, x0:x1]
            cell_xyz = xyz_base[y0:y1, x0:x1]
            cell_depth = depth[y0:y1, x0:x1]
            valid_ratio = np.asarray([cell_valid.mean()], dtype=np.float32)
            if cell_valid.any():
                xyz = cell_xyz[cell_valid]
                d = cell_depth[cell_valid]
                feat = np.concatenate(
                    [
                        xyz.mean(axis=0),
                        xyz.std(axis=0),
                        valid_ratio,
                        np.asarray([d.mean()], dtype=np.float32),
                    ]
                )
            else:
                feat = np.zeros(8, dtype=np.float32)
            features.append(feat.astype(np.float32))
    return np.stack(features, axis=0)


def visible_contact_vector(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    ee_pos: np.ndarray,
) -> np.ndarray:
    xyz_base, valid = backproject_depth(depth, intrinsics, extrinsics)
    workspace = (
        valid
        & (xyz_base[..., 2] > 0.75)
        & (xyz_base[..., 2] < 1.25)
        & (xyz_base[..., 0] > -0.3)
        & (xyz_base[..., 0] < 1.0)
        & (xyz_base[..., 1] > -0.8)
        & (xyz_base[..., 1] < 0.8)
    )
    points = xyz_base[workspace]
    if points.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    ee = np.asarray(ee_pos, dtype=np.float32).reshape(1, 3)
    deltas = points - ee
    return deltas[np.linalg.norm(deltas, axis=1).argmin()].astype(np.float32)


def shuffle_depth_tokens(feature_tokens: np.ndarray, rng: np.random.Generator, num_views: int = 2) -> np.ndarray:
    tokens = feature_tokens.copy()
    tokens_per_view = tokens.shape[0] // num_views
    for view_idx in range(num_views):
        start = view_idx * tokens_per_view
        end = start + tokens_per_view
        tokens[start:end] = tokens[start:end][rng.permutation(tokens_per_view)]
    return tokens


def iter_demo_keys(file_obj: h5py.File) -> Iterable[str]:
    def numeric_suffix(key: str) -> int:
        try:
            return int(key.split("_")[-1])
        except ValueError:
            return 10**9

    return sorted(file_obj["data"].keys(), key=numeric_suffix)


def build_probe_arrays(
    data_dir: Path,
    target_name: str,
    grid_size: int,
    max_samples: int | None,
    stride: int,
    seed: int,
) -> ProbeArrays:
    if target_name not in TARGETS:
        raise ValueError(f"Unknown target {target_name!r}; choose from {TARGETS}")
    rng = np.random.default_rng(seed)
    normal_rows: list[np.ndarray] = []
    null_rows: list[np.ndarray] = []
    shuffle_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []

    for hdf5_path in list_hdf5_files(data_dir):
        with h5py.File(hdf5_path, "r") as f:
            for demo_key in iter_demo_keys(f):
                demo = f["data"][demo_key]
                obs = demo["obs"]
                actions = demo["actions"]
                length = int(actions.shape[0])
                usable_length = length - 1 if target_name == "ee_delta_xyz" else length
                for t in range(0, usable_length, stride):
                    view_tokens = []
                    for view in ("agentview", "eye_in_hand"):
                        view_tokens.append(
                            coarse_geometry_features(
                                obs[f"{view}_depth_m"][t],
                                obs[f"{view}_K"][t],
                                obs[f"{view}_T_camera_to_base"][t],
                                grid_size,
                            )
                        )
                    tokens = np.concatenate(view_tokens, axis=0)
                    aux = np.concatenate(
                        [
                            np.asarray(obs["ee_pos"][t], dtype=np.float32),
                            np.asarray(obs["gripper_states"][t], dtype=np.float32),
                        ]
                    )
                    normal_rows.append(np.concatenate([tokens.reshape(-1), aux]).astype(np.float32))
                    null_rows.append(np.concatenate([np.zeros_like(tokens).reshape(-1), aux]).astype(np.float32))
                    shuffle_rows.append(np.concatenate([shuffle_depth_tokens(tokens, rng).reshape(-1), aux]).astype(np.float32))

                    if target_name == "action_xyz":
                        target = np.asarray(actions[t, :3], dtype=np.float32)
                    elif target_name == "ee_delta_xyz":
                        target = np.asarray(obs["ee_pos"][t + 1] - obs["ee_pos"][t], dtype=np.float32)
                    else:
                        contact = visible_contact_vector(
                            obs["agentview_depth_m"][t],
                            obs["agentview_K"][t],
                            obs["agentview_T_camera_to_base"][t],
                            obs["ee_pos"][t],
                        )
                        target = np.asarray([np.linalg.norm(contact)], dtype=np.float32) if target_name == "contact_distance" else contact
                    target_rows.append(target.astype(np.float32))
                    if max_samples is not None and len(target_rows) >= max_samples:
                        break
                if max_samples is not None and len(target_rows) >= max_samples:
                    break
        if max_samples is not None and len(target_rows) >= max_samples:
            break

    if not target_rows:
        raise RuntimeError(f"No samples built from {data_dir}")
    return ProbeArrays(
        normal=np.stack(normal_rows),
        null=np.stack(null_rows),
        shuffle=np.stack(shuffle_rows),
        target=np.stack(target_rows),
    )


class ProbeMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def standardize(train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, [(arr - mean) / std for arr in arrays], mean, std


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    err = pred - target
    mse = float((err.square().mean()).item())
    mae = float(err.abs().mean().item())
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "rmse": rmse, "mae": mae}


def train_probe(
    arrays: ProbeArrays,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_samples = arrays.target.shape[0]
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = max(1, int(0.8 * num_samples))
    train_idx = indices[:split]
    test_idx = indices[split:]
    if test_idx.size == 0:
        raise RuntimeError("Need at least two samples for a train/test split")

    x_train_raw = arrays.normal[train_idx]
    x_test_normal_raw = arrays.normal[test_idx]
    x_test_null_raw = arrays.null[test_idx]
    x_test_shuffle_raw = arrays.shuffle[test_idx]
    x_train, standardized, _, _ = standardize(x_train_raw, x_test_normal_raw, x_test_null_raw, x_test_shuffle_raw)
    x_test_normal, x_test_null, x_test_shuffle = standardized

    y_train = arrays.target[train_idx]
    y_test = arrays.target[test_idx]
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    y_train_norm = (y_train - y_mean) / y_std

    dataset = TensorDataset(torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train_norm.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ProbeMLP(input_dim=x_train.shape[1], output_dim=y_train.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = F.smooth_l1_loss(pred, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_test_t = torch.from_numpy(y_test.astype(np.float32))
        mean_t = torch.from_numpy(y_mean.astype(np.float32))
        std_t = torch.from_numpy(y_std.astype(np.float32))
        results = {}
        for name, values in {
            "normal": x_test_normal,
            "null": x_test_null,
            "shuffle": x_test_shuffle,
        }.items():
            pred_norm = model(torch.from_numpy(values.astype(np.float32)))
            pred = pred_norm * std_t + mean_t
            results[name] = regression_metrics(pred, y_test_t)

        mean_baseline = torch.from_numpy(np.repeat(y_mean.astype(np.float32), repeats=len(y_test), axis=0))
        results["target_mean_baseline"] = regression_metrics(mean_baseline, y_test_t)

    results["num_samples"] = int(num_samples)
    results["num_train"] = int(len(train_idx))
    results["num_test"] = int(len(test_idx))
    results["input_dim"] = int(arrays.normal.shape[1])
    results["output_dim"] = int(arrays.target.shape[1])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgbd_data_dir", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, default="contact_xyz")
    parser.add_argument("--grid_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    arrays = build_probe_arrays(
        data_dir=args.rgbd_data_dir,
        target_name=args.target,
        grid_size=args.grid_size,
        max_samples=args.max_samples,
        stride=args.stride,
        seed=args.seed,
    )
    results = train_probe(
        arrays=arrays,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    results.update(
        {
            "rgbd_data_dir": str(args.rgbd_data_dir),
            "target": args.target,
            "grid_size": args.grid_size,
            "max_samples": args.max_samples,
            "stride": args.stride,
            "epochs": args.epochs,
        }
    )
    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
