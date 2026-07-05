"""Validate RLBench HDF5 files exported for DepthVLA-OFT."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


REQUIRED_DEMO_KEYS = ("actions", "rlbench_delta_action", "rlbench_keypose_action", "obs")
REQUIRED_OBS_KEYS = (
    "agentview_rgb",
    "eye_in_hand_rgb",
    "agentview_depth_m",
    "eye_in_hand_depth_m",
    "agentview_K",
    "eye_in_hand_K",
    "agentview_T_camera_to_base",
    "eye_in_hand_T_camera_to_base",
    "proprio",
    "ee_pos",
    "rlbench_abs_gripper_pose",
    "rlbench_next_abs_gripper_pose",
)


def finite_ratio(x: np.ndarray) -> float:
    return float(np.isfinite(x).mean()) if x.size else 0.0


def check_shape(name: str, value: np.ndarray, expected_tail: tuple[int, ...]) -> None:
    if value.shape[-len(expected_tail) :] != expected_tail:
        raise ValueError(f"{name} expected trailing shape {expected_tail}, got {value.shape}")


def validate_demo(file_path: Path, demo_key: str, demo, strict: bool) -> dict:
    for key in REQUIRED_DEMO_KEYS:
        if key not in demo:
            raise KeyError(f"{file_path}:{demo_key} missing {key}")
    obs = demo["obs"]
    for key in REQUIRED_OBS_KEYS:
        if key not in obs:
            raise KeyError(f"{file_path}:{demo_key}/obs missing {key}")

    actions = demo["actions"][()]
    delta_actions = demo["rlbench_delta_action"][()]
    keypose_actions = demo["rlbench_keypose_action"][()]
    agent_rgb = obs["agentview_rgb"][()]
    wrist_rgb = obs["eye_in_hand_rgb"][()]
    agent_depth = obs["agentview_depth_m"][()]
    wrist_depth = obs["eye_in_hand_depth_m"][()]
    agent_k = obs["agentview_K"][()]
    wrist_k = obs["eye_in_hand_K"][()]
    agent_t = obs["agentview_T_camera_to_base"][()]
    wrist_t = obs["eye_in_hand_T_camera_to_base"][()]
    proprio = obs["proprio"][()]
    abs_pose = obs["rlbench_abs_gripper_pose"][()]
    next_abs_pose = obs["rlbench_next_abs_gripper_pose"][()]

    length = actions.shape[0]
    if length <= 0:
        raise ValueError(f"{file_path}:{demo_key} has no transitions")
    for name, value in (
        ("rlbench_delta_action", delta_actions),
        ("agentview_rgb", agent_rgb),
        ("eye_in_hand_rgb", wrist_rgb),
        ("agentview_depth_m", agent_depth),
        ("eye_in_hand_depth_m", wrist_depth),
        ("agentview_K", agent_k),
        ("eye_in_hand_K", wrist_k),
        ("agentview_T_camera_to_base", agent_t),
        ("eye_in_hand_T_camera_to_base", wrist_t),
        ("proprio", proprio),
        ("rlbench_abs_gripper_pose", abs_pose),
        ("rlbench_next_abs_gripper_pose", next_abs_pose),
    ):
        if value.shape[0] != length:
            raise ValueError(f"{file_path}:{demo_key}/{name} length {value.shape[0]} != actions length {length}")

    check_shape("actions", actions, (7,))
    check_shape("rlbench_delta_action", delta_actions, (7,))
    check_shape("rlbench_keypose_action", keypose_actions, (8,))
    check_shape("agentview_rgb", agent_rgb, (agent_rgb.shape[-3], agent_rgb.shape[-2], 3))
    check_shape("eye_in_hand_rgb", wrist_rgb, (wrist_rgb.shape[-3], wrist_rgb.shape[-2], 3))
    check_shape("agentview_K", agent_k, (3, 3))
    check_shape("eye_in_hand_K", wrist_k, (3, 3))
    check_shape("agentview_T_camera_to_base", agent_t, (4, 4))
    check_shape("eye_in_hand_T_camera_to_base", wrist_t, (4, 4))
    check_shape("rlbench_abs_gripper_pose", abs_pose, (8,))
    check_shape("rlbench_next_abs_gripper_pose", next_abs_pose, (8,))

    depth_finite = min(finite_ratio(agent_depth), finite_ratio(wrist_depth))
    action_finite = finite_ratio(actions)
    keypose_finite = finite_ratio(keypose_actions)
    if strict:
        if depth_finite < 0.99:
            raise ValueError(f"{file_path}:{demo_key} depth finite ratio too low: {depth_finite:.4f}")
        if action_finite < 1.0 or keypose_finite < 1.0:
            raise ValueError(f"{file_path}:{demo_key} action/keypose contains non-finite values")
        if float(np.abs(actions[:, :6]).max()) > 2.0:
            raise ValueError(f"{file_path}:{demo_key} delta action looks too large: max={np.abs(actions[:, :6]).max()}")

    return {
        "transitions": int(length),
        "agent_depth_min": float(np.nanmin(agent_depth)),
        "agent_depth_max": float(np.nanmax(agent_depth)),
        "wrist_depth_min": float(np.nanmin(wrist_depth)),
        "wrist_depth_max": float(np.nanmax(wrist_depth)),
        "action_abs_max": float(np.abs(actions).max()),
        "keypose_xyz_min": np.nanmin(keypose_actions[:, :3], axis=0),
        "keypose_xyz_max": np.nanmax(keypose_actions[:, :3], axis=0),
    }


def validate_file(file_path: Path, strict: bool) -> dict:
    with h5py.File(file_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{file_path} missing /data group")
        summaries = []
        for demo_key in sorted(f["data"].keys()):
            summaries.append(validate_demo(file_path, demo_key, f["data"][demo_key], strict=strict))
    return {
        "file": str(file_path),
        "num_demos": len(summaries),
        "num_transitions": sum(item["transitions"] for item in summaries),
        "agent_depth_min": min(item["agent_depth_min"] for item in summaries),
        "agent_depth_max": max(item["agent_depth_max"] for item in summaries),
        "wrist_depth_min": min(item["wrist_depth_min"] for item in summaries),
        "wrist_depth_max": max(item["wrist_depth_max"] for item in summaries),
        "action_abs_max": max(item["action_abs_max"] for item in summaries),
        "keypose_xyz_min": np.min(np.stack([item["keypose_xyz_min"] for item in summaries]), axis=0),
        "keypose_xyz_max": np.max(np.stack([item["keypose_xyz_max"] for item in summaries]), axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Directory containing converted RLBench HDF5 files")
    parser.add_argument("--strict", action="store_true", help="Fail on suspicious numeric ranges")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    files = sorted(list(data_dir.glob("*.hdf5")) + list(data_dir.glob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

    total_demos = 0
    total_transitions = 0
    for file_path in files:
        summary = validate_file(file_path, strict=args.strict)
        total_demos += summary["num_demos"]
        total_transitions += summary["num_transitions"]
        print(f"[ok] {file_path.name}")
        print(f"  demos/transitions: {summary['num_demos']} / {summary['num_transitions']}")
        print(f"  agent depth range: {summary['agent_depth_min']:.4f} .. {summary['agent_depth_max']:.4f}")
        print(f"  wrist depth range: {summary['wrist_depth_min']:.4f} .. {summary['wrist_depth_max']:.4f}")
        print(f"  action abs max:    {summary['action_abs_max']:.4f}")
        print(f"  keypose xyz min:   {summary['keypose_xyz_min']}")
        print(f"  keypose xyz max:   {summary['keypose_xyz_max']}")

    print("[done] validation passed")
    print(f"  files:       {len(files)}")
    print(f"  demos:       {total_demos}")
    print(f"  transitions: {total_transitions}")


if __name__ == "__main__":
    main()
