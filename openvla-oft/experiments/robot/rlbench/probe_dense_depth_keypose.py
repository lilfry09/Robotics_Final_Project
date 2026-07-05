"""Offline dense-depth probe for RLBench absolute keypose prediction.

This is a GO/NO-GO gate before expensive VLA training. It trains a small
DensePointDepthTokenEncoder + query head to predict ``rlbench_keypose_action``
from converted RLBench RGB-D HDF5 files, then evaluates the same model with
normal, null, and shuffled geometry.

Success means normal depth predicts the absolute keypose substantially better
than null/shuffled depth. If this fails, scaling RGB-D VLA training is unlikely
to make depth causally useful.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prismatic.models.dense_point_depth_encoder import DensePointDepthTokenEncoder


@dataclass
class ProbeBatchArrays:
    depth_values: np.ndarray
    depth_intrinsics: np.ndarray
    depth_extrinsics: np.ndarray
    depth_valid_mask: np.ndarray
    ee_pos: np.ndarray
    target: np.ndarray


class KeyposeQueryHead(nn.Module):
    """Small query-attention head over dense 3D tokens."""

    def __init__(self, token_dim: int, hidden_dim: int = 256, output_dim: int = 8, num_queries: int = 3) -> None:
        super().__init__()
        num_heads = 8 if token_dim % 8 == 0 else 1
        self.query = nn.Parameter(torch.randn(num_queries, token_dim) * 0.02)
        self.token_norm = nn.LayerNorm(token_dim)
        self.cross_attn = nn.MultiheadAttention(token_dim, num_heads=num_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim * num_queries),
            nn.Linear(token_dim * num_queries, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz = tokens.shape[0]
        queries = self.query.unsqueeze(0).expand(bsz, -1, -1).to(device=tokens.device, dtype=tokens.dtype)
        attended, _ = self.cross_attn(queries, self.token_norm(tokens), self.token_norm(tokens), need_weights=False)
        return self.head(attended.reshape(bsz, -1))


class DenseDepthKeyposeProbe(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, num_points_per_view: int) -> None:
        super().__init__()
        self.encoder = DensePointDepthTokenEncoder(
            llm_dim=token_dim,
            hidden_dim=hidden_dim,
            num_points_per_view=num_points_per_view,
            num_views=2,
            alpha_init=1.0,
        )
        self.head = KeyposeQueryHead(token_dim=token_dim, hidden_dim=hidden_dim, output_dim=8)

    def forward(
        self,
        depth_values: torch.Tensor,
        depth_intrinsics: torch.Tensor,
        depth_extrinsics: torch.Tensor,
        depth_valid_mask: torch.Tensor,
        ee_pos: torch.Tensor,
        ablation_mode: str = "none",
    ) -> torch.Tensor:
        old_mode = self.encoder.ablation_mode
        self.encoder.ablation_mode = ablation_mode
        try:
            tokens = self.encoder(
                depth_values=depth_values,
                depth_intrinsics=depth_intrinsics,
                depth_extrinsics=depth_extrinsics,
                depth_valid_mask=depth_valid_mask,
                ee_pos=ee_pos,
            )
        finally:
            self.encoder.ablation_mode = old_mode
        return self.head(tokens)


def list_hdf5_files(data_dir: Path) -> list[Path]:
    files = sorted(list(data_dir.glob("*.hdf5")) + list(data_dir.glob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")
    return files


def iter_demo_keys(file_obj: h5py.File) -> list[str]:
    def key_order(key: str) -> int:
        try:
            return int(key.split("_")[-1])
        except ValueError:
            return 10**9

    return sorted(file_obj["data"].keys(), key=key_order)


def load_probe_arrays(data_dir: Path, max_samples: int | None, stride: int) -> ProbeBatchArrays:
    depth_values, intrinsics, extrinsics, masks, ee_pos, targets = [], [], [], [], [], []
    for file_path in list_hdf5_files(data_dir):
        with h5py.File(file_path, "r") as f:
            for demo_key in iter_demo_keys(f):
                demo = f["data"][demo_key]
                if "rlbench_keypose_action" not in demo:
                    raise KeyError(f"{file_path}:{demo_key} missing rlbench_keypose_action")
                obs = demo["obs"]
                required = (
                    "agentview_depth_m",
                    "eye_in_hand_depth_m",
                    "agentview_K",
                    "eye_in_hand_K",
                    "agentview_T_camera_to_base",
                    "eye_in_hand_T_camera_to_base",
                    "ee_pos",
                )
                missing = [key for key in required if key not in obs]
                if missing:
                    raise KeyError(f"{file_path}:{demo_key}/obs missing {missing}")
                length = int(demo["rlbench_keypose_action"].shape[0])
                for t in range(0, length, stride):
                    d = np.stack([obs["agentview_depth_m"][t], obs["eye_in_hand_depth_m"][t]], axis=0).astype(np.float32)
                    k = np.stack([obs["agentview_K"][t], obs["eye_in_hand_K"][t]], axis=0).astype(np.float32)
                    e = np.stack(
                        [obs["agentview_T_camera_to_base"][t], obs["eye_in_hand_T_camera_to_base"][t]], axis=0
                    ).astype(np.float32)
                    target = np.asarray(demo["rlbench_keypose_action"][t], dtype=np.float32)
                    if not np.isfinite(d).all() or not np.isfinite(k).all() or not np.isfinite(e).all():
                        continue
                    if not np.isfinite(target).all():
                        continue
                    depth_values.append(d)
                    intrinsics.append(k)
                    extrinsics.append(e)
                    masks.append(np.isfinite(d))
                    ee_pos.append(np.asarray(obs["ee_pos"][t], dtype=np.float32))
                    targets.append(target)
                    if max_samples is not None and len(targets) >= max_samples:
                        break
                if max_samples is not None and len(targets) >= max_samples:
                    break
        if max_samples is not None and len(targets) >= max_samples:
            break

    if not targets:
        raise RuntimeError(f"No valid probe samples found in {data_dir}")
    return ProbeBatchArrays(
        depth_values=np.stack(depth_values),
        depth_intrinsics=np.stack(intrinsics),
        depth_extrinsics=np.stack(extrinsics),
        depth_valid_mask=np.stack(masks),
        ee_pos=np.stack(ee_pos),
        target=np.stack(targets),
    )


def split_indices(num_samples: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_samples)
    if num_samples == 1:
        return order, order
    n_test = max(1, int(round(num_samples * test_fraction)))
    n_test = min(n_test, num_samples - 1)
    return order[n_test:], order[:n_test]


def target_normalize(train_target: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_target.mean(axis=0, keepdims=True)
    std = train_target.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (target - mean) / std, mean, std


def make_tensor_dataset(arrays: ProbeBatchArrays, indices: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray) -> TensorDataset:
    target = (arrays.target[indices] - target_mean) / target_std
    return TensorDataset(
        torch.from_numpy(arrays.depth_values[indices]).float(),
        torch.from_numpy(arrays.depth_intrinsics[indices]).float(),
        torch.from_numpy(arrays.depth_extrinsics[indices]).float(),
        torch.from_numpy(arrays.depth_valid_mask[indices]).bool(),
        torch.from_numpy(arrays.ee_pos[indices]).float(),
        torch.from_numpy(target).float(),
    )


def evaluate(
    model: DenseDepthKeyposeProbe,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    ablation_mode: str,
) -> dict[str, float]:
    model.eval()
    preds, targets = [], []
    with torch.inference_mode():
        for depth, intrinsics, extrinsics, mask, ee_pos, target in loader:
            mode = ablation_mode
            if str(ablation_mode).lower() in ("shuffle_samples", "cross_sample", "batch_shuffle"):
                if depth.shape[0] > 1:
                    perm = torch.arange(depth.shape[0]).roll(1)
                    depth = depth[perm]
                    intrinsics = intrinsics[perm]
                    extrinsics = extrinsics[perm]
                    mask = mask[perm]
                mode = "none"
            pred = model(
                depth.to(device),
                intrinsics.to(device),
                extrinsics.to(device),
                mask.to(device),
                ee_pos.to(device),
                ablation_mode=mode,
            )
            preds.append(pred.cpu())
            targets.append(target.cpu())
    pred = torch.cat(preds, dim=0).numpy() * target_std + target_mean
    target = torch.cat(targets, dim=0).numpy() * target_std + target_mean
    err = pred - target
    xyz_rmse = float(np.sqrt(np.mean(err[:, :3] ** 2)))
    all_rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    return {"rmse": all_rmse, "xyz_rmse": xyz_rmse, "mae": mae}


def run_probe(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    arrays = load_probe_arrays(Path(args.data_dir), max_samples=args.max_samples, stride=args.stride)
    train_idx, test_idx = split_indices(arrays.target.shape[0], args.test_fraction, args.seed)
    _, target_mean, target_std = target_normalize(arrays.target[train_idx], arrays.target[train_idx])
    train_dataset = make_tensor_dataset(arrays, train_idx, target_mean, target_std)
    test_dataset = make_tensor_dataset(arrays, test_idx, target_mean, target_std)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = DenseDepthKeyposeProbe(
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
        num_points_per_view=args.num_points_per_view,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for depth, intrinsics, extrinsics, mask, ee_pos, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(
                depth.to(device),
                intrinsics.to(device),
                extrinsics.to(device),
                mask.to(device),
                ee_pos.to(device),
                ablation_mode="none",
            )
            loss = F.smooth_l1_loss(pred, target.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if args.verbose:
            print(f"epoch {epoch + 1:03d}/{args.epochs}: loss={np.mean(losses):.6f}")

    normal = evaluate(model, test_loader, device, target_mean, target_std, ablation_mode="none")
    null = evaluate(model, test_loader, device, target_mean, target_std, ablation_mode="null")
    shuffle = evaluate(model, test_loader, device, target_mean, target_std, ablation_mode="shuffle_samples")
    results = {
        "data_dir": str(args.data_dir),
        "num_samples": int(arrays.target.shape[0]),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "target": "rlbench_keypose_action",
        "normal": normal,
        "null": null,
        "shuffle": shuffle,
        "advantage_null_xyz_rmse": float(null["xyz_rmse"] - normal["xyz_rmse"]),
        "advantage_shuffle_xyz_rmse": float(shuffle["xyz_rmse"] - normal["xyz_rmse"]),
    }
    return results


def print_results(results: dict, threshold: float) -> None:
    print("=" * 72)
    print("RLBench Dense Depth -> Absolute Keypose Probe")
    print("=" * 72)
    print(f"samples train/test: {results['num_train']} / {results['num_test']}")
    for mode in ("normal", "null", "shuffle"):
        row = results[mode]
        print(f"{mode:8s} rmse={row['rmse']:.6f} xyz_rmse={row['xyz_rmse']:.6f} mae={row['mae']:.6f}")
    print(f"advantage over null xyz_rmse:    {results['advantage_null_xyz_rmse']:+.6f}")
    print(f"advantage over shuffle xyz_rmse: {results['advantage_shuffle_xyz_rmse']:+.6f}")
    if results["advantage_null_xyz_rmse"] >= threshold and results["advantage_shuffle_xyz_rmse"] >= threshold:
        print(f"GO: normal beats null/shuffle by >= {threshold}")
    else:
        print(f"NO-GO: normal does not beat both ablations by >= {threshold}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--num_points_per_view", type=int, default=512)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = run_probe(args)
    print_results(results, threshold=args.threshold)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
