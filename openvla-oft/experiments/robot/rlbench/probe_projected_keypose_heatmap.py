"""Offline RLBench projected-keypose heatmap gate.

This probe tests a stronger spatial-action target than the previous scalar
absolute-keypose auxiliary loss. It trains a small CNN on normal RGB-D geometry
maps to predict where the next absolute gripper keypose projects in each camera
view. The same trained model is evaluated with normal, null, and cross-sample
depth-derived inputs.

GO condition for using this idea in the VLA:

    normal depth must beat null and cross-sample inputs, and predicted heatmaps
    must change when depth is replaced by another sample.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


VIEWS = ("agentview", "eye_in_hand")


@dataclass
class HeatmapArrays:
    inputs: np.ndarray
    targets: np.ndarray
    valid: np.ndarray
    centers_uv: np.ndarray


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


def resize_nearest(array: np.ndarray, out_size: int) -> np.ndarray:
    """Nearest-neighbor resize for small geometry maps without extra deps."""
    if array.shape[0] == out_size and array.shape[1] == out_size:
        return array
    ys = np.linspace(0, array.shape[0] - 1, out_size).round().astype(np.int64)
    xs = np.linspace(0, array.shape[1] - 1, out_size).round().astype(np.int64)
    return array[np.ix_(ys, xs)]


def safe_signed_focal(value: float, eps: float = 1e-6) -> float:
    value = float(value)
    if abs(value) >= eps:
        return value
    return -eps if value < 0 else eps


def backproject_depth(depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = as_depth_2d(depth)
    height, width = depth.shape
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")

    fx = safe_signed_focal(float(intrinsics[0, 0]))
    fy = safe_signed_focal(float(intrinsics[1, 1]))
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    valid = np.isfinite(depth) & (depth >= 0.01) & (depth <= 5.0)
    z_cam = np.where(valid, depth, 0.0).astype(np.float32)
    x_cam = (xs - cx) * z_cam / fx
    y_cam = (ys - cy) * z_cam / fy
    xyz1_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)
    xyz_base = np.einsum("ij,hwj->hwi", extrinsics.astype(np.float32), xyz1_cam)[..., :3]
    return xyz_base.astype(np.float32), valid


def project_point(point_base: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> tuple[float, float, bool]:
    point_base = np.asarray(point_base, dtype=np.float32).reshape(3)
    t_base_to_cam = np.linalg.inv(extrinsics.astype(np.float64)).astype(np.float32)
    point_cam = (t_base_to_cam @ np.asarray([point_base[0], point_base[1], point_base[2], 1.0], dtype=np.float32))[:3]
    if not np.isfinite(point_cam).all() or point_cam[2] <= 0.01:
        return 0.0, 0.0, False
    u = float(intrinsics[0, 0] * point_cam[0] / point_cam[2] + intrinsics[0, 2])
    v = float(intrinsics[1, 1] * point_cam[1] / point_cam[2] + intrinsics[1, 2])
    return u, v, True


def gaussian_heatmap(size: int, u: float, v: float, sigma: float) -> np.ndarray:
    y = np.arange(size, dtype=np.float32)
    x = np.arange(size, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    heatmap = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2.0 * sigma**2))
    max_value = float(heatmap.max())
    if max_value > 1e-8:
        heatmap = heatmap / max_value
    return heatmap.astype(np.float32)


def geometry_map_for_view(obs: h5py.Group, view: str, t: int, map_size: int) -> np.ndarray:
    depth = as_depth_2d(obs[f"{view}_depth_m"][t])
    intrinsics = np.asarray(obs[f"{view}_K"][t], dtype=np.float32)
    extrinsics = np.asarray(obs[f"{view}_T_camera_to_base"][t], dtype=np.float32)
    xyz_base, valid = backproject_depth(depth, intrinsics, extrinsics)

    height, width = depth.shape
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    u_norm = xs / max(width - 1, 1)
    v_norm = ys / max(height - 1, 1)
    z_cam = np.where(valid, depth, 0.0).astype(np.float32)
    valid_f = valid.astype(np.float32)

    channels = np.concatenate(
        [
            xyz_base,
            z_cam[..., None],
            valid_f[..., None],
            u_norm[..., None],
            v_norm[..., None],
        ],
        axis=-1,
    )
    channels = resize_nearest(channels, map_size)
    # CHW for torch. Rotate consistently with prior policy-side convention.
    channels = channels[::-1, ::-1].copy()
    return np.transpose(channels, (2, 0, 1)).astype(np.float32)


def build_arrays(
    data_dir: Path,
    map_size: int,
    sigma: float,
    max_samples: int | None,
    stride: int,
) -> HeatmapArrays:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    centers: list[np.ndarray] = []

    for hdf5_path in iter_hdf5_files(data_dir):
        with h5py.File(hdf5_path, "r") as f:
            for demo_key in iter_demo_keys(f):
                demo = f["data"][demo_key]
                obs = demo["obs"]
                keyposes = demo["rlbench_keypose_action"]
                length = int(keyposes.shape[0])
                for t in range(0, length, stride):
                    keypose_xyz = np.asarray(keyposes[t, :3], dtype=np.float32)
                    if not np.isfinite(keypose_xyz).all():
                        continue

                    view_inputs = []
                    view_targets = []
                    view_valid = []
                    view_centers = []
                    for view in VIEWS:
                        geom = geometry_map_for_view(obs, view, t, map_size)
                        intrinsics = np.asarray(obs[f"{view}_K"][t], dtype=np.float32)
                        extrinsics = np.asarray(obs[f"{view}_T_camera_to_base"][t], dtype=np.float32)
                        u, v, valid = project_point(keypose_xyz, intrinsics, extrinsics)

                        # If the original image is not map_size, scale projected
                        # coordinates into the resized heatmap frame.
                        depth = as_depth_2d(obs[f"{view}_depth_m"][t])
                        scale_u = map_size / float(depth.shape[1])
                        scale_v = map_size / float(depth.shape[0])
                        u_map = u * scale_u
                        v_map = v * scale_v
                        valid = bool(valid and 0.0 <= u_map < map_size and 0.0 <= v_map < map_size)
                        heatmap = gaussian_heatmap(map_size, u_map, v_map, sigma) if valid else np.zeros((map_size, map_size), dtype=np.float32)

                        view_inputs.append(geom)
                        view_targets.append(heatmap)
                        view_valid.append(valid)
                        view_centers.append([u_map, v_map])

                    if not any(view_valid):
                        continue
                    inputs.append(np.concatenate(view_inputs, axis=0))
                    targets.append(np.stack(view_targets, axis=0))
                    valids.append(np.asarray(view_valid, dtype=np.bool_))
                    centers.append(np.asarray(view_centers, dtype=np.float32))
                    if max_samples is not None and len(inputs) >= max_samples:
                        break
                if max_samples is not None and len(inputs) >= max_samples:
                    break
        if max_samples is not None and len(inputs) >= max_samples:
            break

    if not inputs:
        raise RuntimeError(f"No valid projected heatmap samples built from {data_dir}")

    return HeatmapArrays(
        inputs=np.stack(inputs).astype(np.float32),
        targets=np.stack(targets).astype(np.float32),
        valid=np.stack(valids).astype(np.bool_),
        centers_uv=np.stack(centers).astype(np.float32),
    )


class SmallHeatmapNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_mse_from_logits(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    per_view = (pred - target).pow(2).mean(dim=(-1, -2))
    weights = valid.to(per_view.dtype)
    return (per_view * weights).sum() / weights.sum().clamp_min(1.0)


def standardize_inputs(train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    mean = train.mean(axis=(0, 2, 3), keepdims=True)
    std = train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return (train - mean) / std, [(arr - mean) / std for arr in arrays]


def peak_xy(heatmaps: np.ndarray) -> np.ndarray:
    flat = heatmaps.reshape(*heatmaps.shape[:2], -1)
    idx = flat.argmax(axis=-1)
    width = heatmaps.shape[-1]
    y = idx // width
    x = idx % width
    return np.stack([x, y], axis=-1).astype(np.float32)


def evaluate_predictions(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, centers_uv: np.ndarray) -> dict:
    weights = valid.astype(np.float32)
    mse_per_view = ((pred - target) ** 2).mean(axis=(-1, -2))
    mse = float((mse_per_view * weights).sum() / max(float(weights.sum()), 1.0))
    psnr = float(-10.0 * math.log10(max(mse, 1e-12)))
    pred_peak = peak_xy(pred)
    peak_dist = np.linalg.norm(pred_peak - centers_uv, axis=-1)
    peak_dist = float((peak_dist * weights).sum() / max(float(weights.sum()), 1.0))
    return {
        "mse": mse,
        "psnr": psnr,
        "peak_error_px": peak_dist,
    }


def train_and_eval(
    arrays: HeatmapArrays,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    test_fraction: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    num_samples = arrays.inputs.shape[0]
    indices = rng.permutation(num_samples)
    test_size = max(1, int(round(num_samples * test_fraction)))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    if train_idx.shape[0] == 0:
        raise ValueError("Train split is empty; reduce --test_fraction")

    x_train_raw = arrays.inputs[train_idx]
    x_test_normal_raw = arrays.inputs[test_idx]
    null_raw = np.zeros_like(x_test_normal_raw)
    cross_raw = arrays.inputs[rng.permutation(num_samples)[: test_idx.shape[0]]]
    x_train, standardized = standardize_inputs(x_train_raw, x_test_normal_raw, null_raw, cross_raw)
    x_test_normal, x_test_null, x_test_cross = standardized

    y_train = arrays.targets[train_idx]
    valid_train = arrays.valid[train_idx]
    y_test = arrays.targets[test_idx]
    valid_test = arrays.valid[test_idx]
    centers_test = arrays.centers_uv[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallHeatmapNet(in_channels=x_train.shape[1], out_channels=y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_ds = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
        torch.from_numpy(valid_train.astype(np.float32)).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model.train()
    for _ in range(epochs):
        for xb, yb, vb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            vb = vb.to(device).bool()
            loss = masked_mse_from_logits(model(xb), yb, vb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    def predict(x: np.ndarray) -> np.ndarray:
        model.eval()
        outs = []
        with torch.inference_mode():
            for start in range(0, x.shape[0], batch_size):
                xb = torch.from_numpy(x[start : start + batch_size]).float().to(device)
                outs.append(torch.sigmoid(model(xb)).float().cpu().numpy())
        return np.concatenate(outs, axis=0)

    pred_normal = predict(x_test_normal)
    pred_null = predict(x_test_null)
    pred_cross = predict(x_test_cross)
    mean_target = y_train.mean(axis=0, keepdims=True)
    pred_mean = np.broadcast_to(mean_target, y_test.shape)

    return {
        "num_samples": int(num_samples),
        "train_samples": int(train_idx.shape[0]),
        "test_samples": int(test_idx.shape[0]),
        "valid_view_fraction": float(arrays.valid.mean()),
        "normal": evaluate_predictions(pred_normal, y_test, valid_test, centers_test),
        "null": evaluate_predictions(pred_null, y_test, valid_test, centers_test),
        "cross_sample": evaluate_predictions(pred_cross, y_test, valid_test, centers_test),
        "mean_baseline": evaluate_predictions(pred_mean, y_test, valid_test, centers_test),
        "paired_delta": {
            "normal_vs_null_l1": float(np.mean(np.abs(pred_normal - pred_null))),
            "normal_vs_cross_l1": float(np.mean(np.abs(pred_normal - pred_cross))),
            "normal_vs_null_peak_l2": float(np.linalg.norm(peak_xy(pred_normal) - peak_xy(pred_null), axis=-1).mean()),
            "normal_vs_cross_peak_l2": float(np.linalg.norm(peak_xy(pred_normal) - peak_xy(pred_cross), axis=-1).mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgbd_data_dir", type=Path, required=True)
    parser.add_argument("--map_size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--test_fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_json", type=Path, default=Path("experiments/logs/rlbench_projected_keypose_heatmap_probe.json"))
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    arrays = build_arrays(
        data_dir=args.rgbd_data_dir,
        map_size=args.map_size,
        sigma=args.sigma,
        max_samples=args.max_samples,
        stride=args.stride,
    )
    result = {
        "config": {
            "rgbd_data_dir": str(args.rgbd_data_dir),
            "map_size": args.map_size,
            "sigma": args.sigma,
            "max_samples": args.max_samples,
            "stride": args.stride,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "result": train_and_eval(
            arrays=arrays,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            test_fraction=args.test_fraction,
            seed=args.seed,
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[done] wrote {args.output_json}")


if __name__ == "__main__":
    main()
