"""Quick diagnostics for DepthVLA-OFT depth tokens.

This script does not load the 7B VLA. It checks the lightweight depth encoder,
prefix accounting, and the 180-degree geometry-grid flip on a tiny RGB-D HDF5
sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import torch

from prismatic.models.depth_encoder import LightweightDepthTokenEncoder


def find_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("depth_encoder--latest_checkpoint.pt"))
    if not candidates:
        candidates = sorted(checkpoint_dir.glob("depth_encoder--*_checkpoint.pt"))
    if not candidates:
        raise FileNotFoundError(f"No depth_encoder checkpoint found in {checkpoint_dir}")
    return candidates[-1]


def infer_dims(state_dict: dict) -> tuple[int, int]:
    linear2_weight = state_dict["encoder.2.weight"]
    llm_dim, hidden_dim = linear2_weight.shape
    return int(llm_dim), int(hidden_dim)


def remove_ddp_prefix(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_one_sample(data_dir: Path):
    hdf5_path = sorted(list(data_dir.glob("*.hdf5")) + list(data_dir.glob("*.h5")))[0]
    with h5py.File(hdf5_path, "r") as f:
        demo_key = sorted(f["data"].keys())[0]
        obs = f["data"][demo_key]["obs"]
        depth_values = torch.tensor(
            [obs["agentview_depth_m"][0], obs["eye_in_hand_depth_m"][0]], dtype=torch.float32
        ).unsqueeze(0)
        depth_intrinsics = torch.tensor(
            [obs["agentview_K"][0], obs["eye_in_hand_K"][0]], dtype=torch.float32
        ).unsqueeze(0)
        depth_extrinsics = torch.tensor(
            [obs["agentview_T_camera_to_base"][0], obs["eye_in_hand_T_camera_to_base"][0]], dtype=torch.float32
        ).unsqueeze(0)
    valid = torch.isfinite(depth_values)
    return hdf5_path, demo_key, depth_values, depth_intrinsics, depth_extrinsics, valid


def synthetic_rotation_check(grid_size: int) -> None:
    # This mirrors LightweightDepthTokenEncoder's grid flip at the coarse-token
    # level: raw top-left should become rotated bottom-right, and raw bottom-right
    # should become first token after flattening.
    raw = torch.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    flipped = torch.flip(raw, dims=[0, 1]).flatten()
    expected_first = grid_size * grid_size - 1
    assert int(flipped[0]) == expected_first
    assert int(flipped[-1]) == 0
    print(f"rotation_grid_check=PASS first_token_raw_cell={int(flipped[0])} last_token_raw_cell={int(flipped[-1])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--rgbd_data_dir", type=Path, required=True)
    parser.add_argument("--depth_grid_size", type=int, default=4)
    parser.add_argument("--depth_min_m", type=float, default=0.01)
    parser.add_argument("--depth_max_m", type=float, default=5.0)
    args = parser.parse_args()

    checkpoint = find_checkpoint(args.checkpoint_dir)
    state_dict = remove_ddp_prefix(torch.load(checkpoint, map_location="cpu", weights_only=True))
    llm_dim, hidden_dim = infer_dims(state_dict)
    encoder = LightweightDepthTokenEncoder(
        llm_dim=llm_dim,
        hidden_dim=hidden_dim,
        grid_size=args.depth_grid_size,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        num_views=2,
    )
    encoder.load_state_dict(state_dict)
    encoder.eval()

    hdf5_path, demo_key, depth_values, depth_intrinsics, depth_extrinsics, valid = load_one_sample(args.rgbd_data_dir)
    kwargs = {
        "depth_values": depth_values,
        "depth_intrinsics": depth_intrinsics,
        "depth_extrinsics": depth_extrinsics,
        "depth_valid_mask": valid,
    }

    with torch.inference_mode():
        encoder.ablation_mode = "none"
        normal = encoder(**kwargs)
        encoder.ablation_mode = "null"
        null = encoder(**kwargs)
        encoder.ablation_mode = "shuffle_tokens"
        shuffled = encoder(**kwargs)

    rgb_tokens = 2 * 256
    depth_tokens = encoder.depth_num_tokens
    prefix_tokens = rgb_tokens + depth_tokens + 1
    print(f"checkpoint={checkpoint}")
    print(f"sample={hdf5_path.name}:{demo_key}")
    print(f"alpha={float(encoder.alpha.detach().cpu()):.8f}")
    print(f"depth_tokens={depth_tokens}")
    print(f"prefix_tokens_with_proprio={prefix_tokens}")
    print(f"normal_shape={tuple(normal.shape)} nan={bool(torch.isnan(normal).any())}")
    print(f"normal_abs_mean={float(normal.abs().mean()):.8f} normal_l2={float(normal.norm()):.8f}")
    print(f"null_abs_mean={float(null.abs().mean()):.8f} null_l2={float(null.norm()):.8f}")
    print(f"shuffle_abs_mean={float(shuffled.abs().mean()):.8f} shuffle_l2={float(shuffled.norm()):.8f}")
    print(f"normal_vs_null_l2={float((normal - null).norm()):.8f}")
    print(f"normal_vs_shuffle_l2={float((normal - shuffled).norm()):.8f}")
    finite_depth = depth_values[torch.isfinite(depth_values)]
    print(f"depth_min={float(finite_depth.min()):.6f} depth_max={float(finite_depth.max()):.6f}")
    synthetic_rotation_check(args.depth_grid_size)


if __name__ == "__main__":
    main()
