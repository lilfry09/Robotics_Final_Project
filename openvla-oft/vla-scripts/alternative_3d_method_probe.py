"""Offline probes for PointVLA/SpatialVLA/BridgeVLA-style 3D representations.

This is not a full paper reproduction. It is a cheap, controlled proxy that
answers the next useful question for this repo:

    Do paper-inspired 3D representations separate normal depth from null or
    corrupted depth on task-relevant labels before we spend time on rollouts?

Implemented proxies:

- bridge2d: BridgeVLA-style 3D-to-2D geometry maps.
- pointvla: PointVLA-style point-cloud set encoder.
- spatialvla: SpatialVLA-style end-effector-relative Ego3D grid features.

Each method is trained on normal depth and evaluated with normal/null/corrupt
depth under the same train/test split.
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


TARGETS = ("task_3d", "action_xyz", "ee_to_object_xyz", "object_to_target_xyz", "gripper_to_contact_distance")
METHODS = ("bridge2d", "pointvla", "spatialvla")
VIEWS = ("agentview", "eye_in_hand")


@dataclass
class MethodArrays:
    normal: np.ndarray
    null: np.ndarray
    corrupt: np.ndarray
    target: np.ndarray


def iter_hdf5_files(data_dir: Path) -> list[Path]:
    files = sorted(list(data_dir.glob("*.hdf5")) + list(data_dir.glob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")
    return files


def iter_demo_keys(file_obj: h5py.File) -> Iterable[str]:
    def numeric_suffix(key: str) -> int:
        try:
            return int(key.split("_")[-1])
        except ValueError:
            return 10**9

    return sorted(file_obj["data"].keys(), key=numeric_suffix)


def as_depth_2d(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth map with shape (H,W) or (H,W,1), got {depth.shape}")
    return depth


def backproject_depth(depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = as_depth_2d(depth)
    height, width = depth.shape
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    fx = fx if abs(fx) >= 1e-6 else (-1e-6 if fx < 0 else 1e-6)
    fy = fy if abs(fy) >= 1e-6 else (-1e-6 if fy < 0 else 1e-6)
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    valid = np.isfinite(depth) & (depth >= 0.01) & (depth <= 5.0)
    z_cam = np.where(valid, depth, 0.0).astype(np.float32)
    x_cam = (xs - cx) * z_cam / fx
    y_cam = (ys - cy) * z_cam / fy
    xyz1_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)
    xyz_base = np.einsum("ij,hwj->hwi", extrinsics.astype(np.float32), xyz1_cam)[..., :3]
    return xyz_base.astype(np.float32), valid


def pool_cell_features(features: np.ndarray, valid: np.ndarray, grid_size: int) -> np.ndarray:
    height, width = valid.shape
    rows = []
    for gy in range(grid_size):
        y0 = round(gy * height / grid_size)
        y1 = round((gy + 1) * height / grid_size)
        for gx in range(grid_size):
            x0 = round(gx * width / grid_size)
            x1 = round((gx + 1) * width / grid_size)
            cell_valid = valid[y0:y1, x0:x1]
            cell = features[y0:y1, x0:x1]
            valid_ratio = np.asarray([cell_valid.mean()], dtype=np.float32)
            if cell_valid.any():
                mean = cell[cell_valid].mean(axis=0).astype(np.float32)
            else:
                mean = np.zeros(features.shape[-1], dtype=np.float32)
            rows.append(np.concatenate([mean, valid_ratio], axis=0))
    return np.stack(rows, axis=0).astype(np.float32)


def bridge2d_features(obs: h5py.Group, t: int, map_size: int) -> np.ndarray:
    """BridgeVLA proxy: multi-view 2D maps with per-cell 3D geometry channels."""
    view_maps = []
    for view_idx, view in enumerate(VIEWS):
        depth = as_depth_2d(obs[f"{view}_depth_m"][t])
        xyz, valid = backproject_depth(depth, obs[f"{view}_K"][t], obs[f"{view}_T_camera_to_base"][t])
        height, width = depth.shape
        ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
        u_norm = xs / max(width - 1, 1)
        v_norm = ys / max(height - 1, 1)
        z_cam = np.where(valid, depth, 0.0).astype(np.float32)
        channels = np.concatenate(
            [
                xyz,
                z_cam[..., None],
                u_norm[..., None],
                v_norm[..., None],
                np.full((*depth.shape, 1), float(view_idx), dtype=np.float32),
            ],
            axis=-1,
        )
        pooled = pool_cell_features(channels, valid, map_size)
        # Match the policy-side 180-degree LIBERO RGB rotation convention.
        pooled = pooled.reshape(map_size, map_size, -1)[::-1, ::-1].reshape(map_size * map_size, -1)
        view_maps.append(pooled)
    ee = np.asarray(obs["ee_pos"][t], dtype=np.float32).reshape(-1)[:3]
    gripper = np.asarray(obs["gripper_states"][t], dtype=np.float32).reshape(-1)
    return np.concatenate([np.concatenate(view_maps, axis=0).reshape(-1), ee, gripper]).astype(np.float32)


def spatialvla_features(obs: h5py.Group, t: int, grid_size: int) -> np.ndarray:
    """SpatialVLA proxy: EE-relative Ego3D grid features."""
    ee = np.asarray(obs["ee_pos"][t], dtype=np.float32).reshape(1, 3)
    rows = []
    for view_idx, view in enumerate(VIEWS):
        depth = as_depth_2d(obs[f"{view}_depth_m"][t])
        xyz, valid = backproject_depth(depth, obs[f"{view}_K"][t], obs[f"{view}_T_camera_to_base"][t])
        dxyz = xyz - ee
        radius = np.linalg.norm(dxyz, axis=-1, keepdims=True).astype(np.float32)
        height, width = depth.shape
        ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
        u_norm = xs / max(width - 1, 1)
        v_norm = ys / max(height - 1, 1)
        z_cam = np.where(valid, depth, 0.0).astype(np.float32)
        features = np.concatenate(
            [
                dxyz,
                radius,
                z_cam[..., None],
                u_norm[..., None],
                v_norm[..., None],
                np.full((*depth.shape, 1), float(view_idx), dtype=np.float32),
            ],
            axis=-1,
        )
        pooled = pool_cell_features(features, valid, grid_size)
        pooled = pooled.reshape(grid_size, grid_size, -1)[::-1, ::-1].reshape(grid_size * grid_size, -1)
        rows.append(pooled)
    gripper = np.asarray(obs["gripper_states"][t], dtype=np.float32).reshape(-1)
    return np.concatenate([np.concatenate(rows, axis=0).reshape(-1), gripper]).astype(np.float32)


def pointvla_points(obs: h5py.Group, t: int, num_points: int, rng: np.random.Generator) -> np.ndarray:
    """PointVLA proxy: fixed-size EE-relative point set from both depth views."""
    ee = np.asarray(obs["ee_pos"][t], dtype=np.float32).reshape(1, 3)
    all_points = []
    for view_idx, view in enumerate(VIEWS):
        depth = as_depth_2d(obs[f"{view}_depth_m"][t])
        xyz, valid = backproject_depth(depth, obs[f"{view}_K"][t], obs[f"{view}_T_camera_to_base"][t])
        height, width = depth.shape
        ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
        # Conservative tabletop mask keeps the point budget focused on manipulation-relevant geometry.
        workspace = (
            valid
            & (xyz[..., 2] > 0.65)
            & (xyz[..., 2] < 1.35)
            & (xyz[..., 0] > -0.5)
            & (xyz[..., 0] < 1.2)
            & (xyz[..., 1] > -1.0)
            & (xyz[..., 1] < 1.0)
        )
        dxyz = xyz - ee
        z_cam = np.where(valid, depth, 0.0).astype(np.float32)
        feats = np.concatenate(
            [
                dxyz,
                np.linalg.norm(dxyz, axis=-1, keepdims=True).astype(np.float32),
                z_cam[..., None],
                (xs / max(width - 1, 1))[..., None],
                (ys / max(height - 1, 1))[..., None],
                np.full((*depth.shape, 1), float(view_idx), dtype=np.float32),
            ],
            axis=-1,
        )
        pts = feats[workspace]
        if pts.shape[0] > 0:
            all_points.append(pts.astype(np.float32))
    if all_points:
        pts = np.concatenate(all_points, axis=0)
        replace = pts.shape[0] < num_points
        idx = rng.choice(pts.shape[0], size=num_points, replace=replace)
        return pts[idx].astype(np.float32)
    return np.zeros((num_points, 8), dtype=np.float32)


def corrupt_flat_geometry(features: np.ndarray, rng: np.random.Generator, tail_dim: int = 0) -> np.ndarray:
    core = features[:-tail_dim] if tail_dim else features
    tail = features[-tail_dim:] if tail_dim else np.zeros((0,), dtype=np.float32)
    if core.size == 0:
        return features.copy()
    # Shuffle coarse cells while preserving each cell's channel vector.
    # This approximates Bridge/Spatial corruption without changing feature scale.
    channel_dim_candidates = (8, 9)
    channel_dim = next((c for c in channel_dim_candidates if core.size % c == 0), None)
    if channel_dim is None:
        shuffled = core.copy()
        rng.shuffle(shuffled)
        return np.concatenate([shuffled, tail]).astype(np.float32)
    cells = core.reshape(-1, channel_dim)
    shuffled = cells[rng.permutation(cells.shape[0])].reshape(-1)
    return np.concatenate([shuffled, tail]).astype(np.float32)


def target_from_obs(obs: h5py.Group, actions: h5py.Dataset, t: int, target_name: str) -> np.ndarray:
    if target_name == "action_xyz":
        return np.asarray(actions[t, :3], dtype=np.float32)
    if target_name == "ee_to_object_xyz":
        return np.asarray(obs["ee_to_object_xyz"][t], dtype=np.float32).reshape(3)
    if target_name == "object_to_target_xyz":
        return np.asarray(obs["object_to_target_xyz"][t], dtype=np.float32).reshape(3)
    if target_name == "gripper_to_contact_distance":
        return np.asarray(obs["gripper_to_contact_distance"][t], dtype=np.float32).reshape(1)
    if target_name == "task_3d":
        return np.concatenate(
            [
                np.asarray(obs["ee_to_object_xyz"][t], dtype=np.float32).reshape(3),
                np.asarray(obs["object_to_target_xyz"][t], dtype=np.float32).reshape(3),
                np.asarray(obs["gripper_to_contact_distance"][t], dtype=np.float32).reshape(1),
            ]
        ).astype(np.float32)
    raise ValueError(f"Unknown target: {target_name}")


def build_arrays(
    data_dir: Path,
    method: str,
    target_name: str,
    max_samples: int | None,
    stride: int,
    seed: int,
    grid_size: int,
    map_size: int,
    num_points: int,
) -> MethodArrays:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if target_name not in TARGETS:
        raise ValueError(f"Unknown target {target_name!r}; choose from {TARGETS}")
    rng = np.random.default_rng(seed)
    normal_rows, null_rows, target_rows = [], [], []

    for hdf5_path in iter_hdf5_files(data_dir):
        with h5py.File(hdf5_path, "r") as f:
            for demo_key in iter_demo_keys(f):
                demo = f["data"][demo_key]
                obs = demo["obs"]
                actions = demo["actions"]
                for t in range(0, int(actions.shape[0]), stride):
                    if method == "bridge2d":
                        feat = bridge2d_features(obs, t, map_size)
                        null = np.zeros_like(feat)
                    elif method == "spatialvla":
                        feat = spatialvla_features(obs, t, grid_size)
                        null = np.zeros_like(feat)
                    else:
                        feat = pointvla_points(obs, t, num_points, rng)
                        null = np.zeros_like(feat)
                    target = target_from_obs(obs, actions, t, target_name)
                    if not np.isfinite(feat).all() or not np.isfinite(target).all():
                        continue
                    normal_rows.append(feat.astype(np.float32))
                    null_rows.append(null.astype(np.float32))
                    target_rows.append(target.astype(np.float32))
                    if max_samples is not None and len(target_rows) >= max_samples:
                        break
                if max_samples is not None and len(target_rows) >= max_samples:
                    break
        if max_samples is not None and len(target_rows) >= max_samples:
            break

    if not target_rows:
        raise RuntimeError(f"No samples built from {data_dir}")

    normal = np.stack(normal_rows).astype(np.float32)
    null = np.stack(null_rows).astype(np.float32)
    target = np.stack(target_rows).astype(np.float32)
    if method == "pointvla":
        # PointNet is order-invariant, so token shuffling is not a meaningful corruption.
        # Use mismatched point clouds from other timesteps instead.
        perm = rng.permutation(normal.shape[0])
        corrupt = normal[perm].copy()
    elif method == "bridge2d":
        corrupt = np.stack([corrupt_flat_geometry(row, rng, tail_dim=5) for row in normal]).astype(np.float32)
    else:
        corrupt = np.stack([corrupt_flat_geometry(row, rng, tail_dim=2) for row in normal]).astype(np.float32)
    return MethodArrays(normal=normal, null=null, corrupt=corrupt, target=target)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointNetRegressor(nn.Module):
    def __init__(self, point_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        emb = self.point_mlp(points)
        pooled = torch.cat([emb.mean(dim=1), emb.max(dim=1).values], dim=-1)
        return self.head(pooled)


def standardize_train(train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    axes = tuple(range(train.ndim - 1))
    mean = train.mean(axis=axes, keepdims=True)
    std = train.std(axis=axes, keepdims=True) + 1e-6
    return (train - mean) / std, [(arr - mean) / std for arr in arrays], mean, std


def train_and_eval(
    arrays: MethodArrays,
    method: str,
    seed: int,
    test_fraction: float,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
) -> dict:
    rng = np.random.default_rng(seed)
    num_samples = arrays.target.shape[0]
    indices = rng.permutation(num_samples)
    test_size = max(1, int(round(num_samples * test_fraction)))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    if train_idx.shape[0] == 0:
        raise ValueError("Train split is empty; reduce --test_fraction")

    x_train = arrays.normal[train_idx]
    x_test_normal = arrays.normal[test_idx]
    x_test_null = arrays.null[test_idx]
    x_test_corrupt = arrays.corrupt[test_idx]
    y_train = arrays.target[train_idx]
    y_test = arrays.target[test_idx]

    x_train, standardized, _, _ = standardize_train(x_train, x_test_normal, x_test_null, x_test_corrupt)
    x_test_normal, x_test_null, x_test_corrupt = standardized
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-6
    y_train_std = (y_train - y_mean) / y_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if method == "pointvla":
        model = PointNetRegressor(x_train.shape[-1], y_train.shape[-1], hidden_dim).to(device)
    else:
        model = MLPRegressor(int(np.prod(x_train.shape[1:])), y_train.shape[-1], hidden_dim).to(device)
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test_normal = x_test_normal.reshape(x_test_normal.shape[0], -1)
        x_test_null = x_test_null.reshape(x_test_null.shape[0], -1)
        x_test_corrupt = x_test_corrupt.reshape(x_test_corrupt.shape[0], -1)

    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train_std).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def predict(x: np.ndarray) -> np.ndarray:
        model.eval()
        outs = []
        with torch.inference_mode():
            for start in range(0, x.shape[0], batch_size):
                xb = torch.from_numpy(x[start : start + batch_size]).float().to(device)
                outs.append(model(xb).float().cpu().numpy())
        pred = np.concatenate(outs, axis=0)
        return pred * y_std + y_mean

    pred_normal = predict(x_test_normal)
    pred_null = predict(x_test_null)
    pred_corrupt = predict(x_test_corrupt)
    baseline = np.broadcast_to(y_train.mean(axis=0, keepdims=True), y_test.shape)

    def metrics(pred: np.ndarray) -> dict:
        err = pred - y_test
        return {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
        }

    return {
        "num_samples": int(num_samples),
        "train_samples": int(train_idx.shape[0]),
        "test_samples": int(test_idx.shape[0]),
        "normal": metrics(pred_normal),
        "null": metrics(pred_null),
        "corrupt": metrics(pred_corrupt),
        "mean_baseline": metrics(baseline),
        "action_delta": {
            "normal_vs_null_l2": float(np.linalg.norm(pred_normal - pred_null, axis=-1).mean()),
            "normal_vs_corrupt_l2": float(np.linalg.norm(pred_normal - pred_corrupt, axis=-1).mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgbd_data_dir", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="bridge2d,pointvla,spatialvla")
    parser.add_argument("--target", choices=TARGETS, default="task_3d")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--grid_size", type=int, default=4)
    parser.add_argument("--map_size", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--test_fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    selected = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(selected) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; choose from {METHODS}")

    results = {
        "config": {
            "rgbd_data_dir": str(args.rgbd_data_dir),
            "methods": selected,
            "target": args.target,
            "max_samples": args.max_samples,
            "stride": args.stride,
            "grid_size": args.grid_size,
            "map_size": args.map_size,
            "num_points": args.num_points,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "methods": {},
    }

    for method in selected:
        print(f"\n=== {method} -> {args.target} ===")
        arrays = build_arrays(
            data_dir=args.rgbd_data_dir,
            method=method,
            target_name=args.target,
            max_samples=args.max_samples,
            stride=args.stride,
            seed=args.seed,
            grid_size=args.grid_size,
            map_size=args.map_size,
            num_points=args.num_points,
        )
        result = train_and_eval(
            arrays=arrays,
            method=method,
            seed=args.seed,
            test_fraction=args.test_fraction,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
        )
        results["methods"][method] = result
        print(json.dumps(result, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        print(f"\nSaved results to: {args.output_json}")


if __name__ == "__main__":
    main()
