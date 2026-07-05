"""Convert RLBench/PerAct-style demonstrations to DepthVLA-compatible HDF5.

The output intentionally mirrors the LIBERO RGB-D HDF5 layout used by
``vla-scripts/finetune_depthvla.py`` while also storing absolute gripper
keypose labels for the next RGB-D experiment.

Expected input layout, matching common PerAct/RVT/BridgeVLA RLBench exports:

    DATA_ROOT/
      train/<task>/all_variations/episodes/episode0/...
      val/<task>/all_variations/episodes/episode0/...

or:

    DATA_ROOT/
      <task>/all_variations/episodes/episode0/...

The script loads episodes through ``peract_colab.rlbench.utils.get_stored_demo``
when available. That helper reconstructs Observation objects with RGB, depth,
point cloud, camera matrices, and low-dimensional state.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover - environment setup guard
    raise ImportError("h5py is required. Install it in the training environment.") from exc


DEFAULT_TASKS = (
    "slide_block_to_target",
    "turn_tap",
    "close_jar",
    "open_drawer",
    "reach_target",
    "pick_up_cup",
)

CAMERA_ALIASES = {
    "front": "agentview",
    "left_shoulder": "agentview",
    "right_shoulder": "agentview",
    "wrist": "eye_in_hand",
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_stored_demo(episodes_dir: Path, episode_idx: int):
    try:
        from peract_colab.rlbench.utils import get_stored_demo
    except ImportError as exc:
        raise ImportError(
            "Could not import peract_colab.rlbench.utils.get_stored_demo. "
            "Install RLBench/PyRep/peract helpers first, or use BridgeVLA's RLBench setup."
        ) from exc
    return get_stored_demo(data_path=str(episodes_dir), index=episode_idx)


def read_variation_description(episode_dir: Path) -> str:
    desc_path = episode_dir / "variation_descriptions.pkl"
    if not desc_path.exists():
        return episode_dir.parent.parent.parent.name.replace("_", " ")
    with desc_path.open("rb") as f:
        descs = pickle.load(f)
    if isinstance(descs, (list, tuple)) and descs:
        return str(descs[0])
    return str(descs)


def obs_value(obs, name: str):
    if hasattr(obs, name):
        return getattr(obs, name)
    misc = getattr(obs, "misc", None)
    if isinstance(misc, dict) and name in misc:
        return misc[name]
    return None


def camera_value(obs, camera: str, suffix: str):
    direct = obs_value(obs, f"{camera}_{suffix}")
    if direct is not None:
        return direct
    if suffix == "camera_intrinsics":
        return obs_value(obs, f"{camera}_camera_intrinsics")
    if suffix == "camera_extrinsics":
        return obs_value(obs, f"{camera}_camera_extrinsics")
    return None


def ensure_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb.shape}")
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = np.clip(rgb * 255.0, 0, 255)
        rgb = rgb.astype(np.uint8)
    return rgb


def ensure_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth image with shape HxW, got {depth.shape}")
    return depth


def ensure_matrix(value: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray:
    if value is None:
        raise KeyError(f"Missing camera matrix: {name}")
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"Expected {name} shape {shape}, got {value.shape}")
    return value


def quat_inverse(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    out = q.copy()
    out[:3] *= -1.0
    denom = float(np.dot(q, q))
    if denom <= 1e-8:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    return out / denom


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float32)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def gripper_pose(obs) -> np.ndarray:
    pose = obs_value(obs, "gripper_pose")
    if pose is None:
        raise KeyError("Observation is missing gripper_pose")
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[0] < 7:
        raise ValueError(f"Expected gripper_pose with at least 7 dims, got {pose.shape}")
    return pose[:7]


def gripper_open(obs) -> float:
    value = obs_value(obs, "gripper_open")
    if value is None:
        return 1.0
    return float(value)


def low_dim_proprio(obs, step_idx: int, episode_len: int) -> np.ndarray:
    pose = gripper_pose(obs)
    gripper = np.array([gripper_open(obs)], dtype=np.float32)
    # Keep this 8-D to match the current OpenVLA-OFT PROPRIO_DIM used by the
    # LIBERO pipeline: Cartesian EE position, EE quaternion, gripper open.
    # Joint positions can be added later with a platform-specific constant.
    return np.concatenate([pose[:3], pose[3:7], gripper]).astype(np.float32)


def delta_action(current_obs, next_obs) -> np.ndarray:
    pose_t = gripper_pose(current_obs)
    pose_tp1 = gripper_pose(next_obs)
    delta_xyz = pose_tp1[:3] - pose_t[:3]
    rel_quat = quat_multiply(pose_tp1[3:7], quat_inverse(pose_t[3:7]))
    delta_rpy = quat_to_euler_xyz(rel_quat)
    return np.concatenate([delta_xyz, delta_rpy, np.array([gripper_open(next_obs)], dtype=np.float32)]).astype(np.float32)


def abs_keypose_action(next_obs) -> np.ndarray:
    pose = gripper_pose(next_obs)
    return np.concatenate([pose, np.array([gripper_open(next_obs)], dtype=np.float32)]).astype(np.float32)


def task_episodes_dir(data_root: Path, split: str, task: str) -> Path:
    candidates = (
        data_root / split / task / "all_variations" / "episodes",
        data_root / task / "all_variations" / "episodes",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find RLBench episodes for task={task!r}; tried {candidates}")


def episode_indices(episodes_dir: Path, max_demos: int | None) -> list[int]:
    indices = []
    for path in sorted(episodes_dir.glob("episode*")):
        suffix = path.name.replace("episode", "")
        if suffix.isdigit():
            indices.append(int(suffix))
    if max_demos is not None:
        indices = indices[:max_demos]
    return indices


def export_task(
    data_root: Path,
    target_dir: Path,
    split: str,
    task: str,
    cameras: tuple[str, str],
    max_demos: int | None,
    overwrite: bool,
) -> Path:
    episodes_dir = task_episodes_dir(data_root, split, task)
    indices = episode_indices(episodes_dir, max_demos)
    if not indices:
        raise FileNotFoundError(f"No episode folders found in {episodes_dir}")

    out_path = target_dir / f"rlbench_{split}_{task}.hdf5"
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} exists; pass --overwrite to regenerate")
        return out_path

    target_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        data_grp = h5.create_group("data")
        h5.attrs["source"] = "rlbench"
        h5.attrs["split"] = split
        h5.attrs["task_name"] = task
        h5.attrs["camera_0"] = cameras[0]
        h5.attrs["camera_1"] = cameras[1]

        for out_idx, ep_idx in enumerate(indices):
            episode_dir = episodes_dir / f"episode{ep_idx}"
            description = read_variation_description(episode_dir)
            demo = load_stored_demo(episodes_dir, ep_idx)
            if len(demo) < 2:
                print(f"[warn] skip task={task} episode={ep_idx}: too short")
                continue

            length = len(demo) - 1
            ep_grp = data_grp.create_group(f"demo_{out_idx}")
            ep_grp.attrs["task_name"] = task
            ep_grp.attrs["language_instruction"] = description
            ep_grp.attrs["source_episode"] = int(ep_idx)

            actions = []
            keypose_actions = []
            abs_poses = []
            next_abs_poses = []
            proprios = []
            obs_buffers: dict[str, list[np.ndarray]] = {
                "agentview_rgb": [],
                "eye_in_hand_rgb": [],
                "agentview_depth_m": [],
                "eye_in_hand_depth_m": [],
                "agentview_K": [],
                "eye_in_hand_K": [],
                "agentview_T_camera_to_base": [],
                "eye_in_hand_T_camera_to_base": [],
                "agentview_point_cloud": [],
                "eye_in_hand_point_cloud": [],
            }

            for t in range(length):
                obs = demo[t]
                next_obs = demo[t + 1]
                actions.append(delta_action(obs, next_obs))
                keypose_actions.append(abs_keypose_action(next_obs))
                abs_poses.append(abs_keypose_action(obs))
                next_abs_poses.append(abs_keypose_action(next_obs))
                proprios.append(low_dim_proprio(obs, t, length))

                for camera in cameras:
                    alias = CAMERA_ALIASES.get(camera, camera)
                    if alias not in ("agentview", "eye_in_hand"):
                        raise ValueError(f"Unsupported camera alias {alias!r}; expected agentview/eye_in_hand mapping")
                    rgb = ensure_rgb(camera_value(obs, camera, "rgb"))
                    depth = ensure_depth(camera_value(obs, camera, "depth"))
                    intrinsics = ensure_matrix(
                        camera_value(obs, camera, "camera_intrinsics"),
                        (3, 3),
                        f"{camera}_camera_intrinsics",
                    )
                    extrinsics = ensure_matrix(
                        camera_value(obs, camera, "camera_extrinsics"),
                        (4, 4),
                        f"{camera}_camera_extrinsics",
                    )
                    point_cloud = camera_value(obs, camera, "point_cloud")
                    if point_cloud is None:
                        point_cloud = np.zeros((*depth.shape, 3), dtype=np.float32)
                    point_cloud = np.asarray(point_cloud, dtype=np.float32)
                    obs_buffers[f"{alias}_rgb"].append(rgb)
                    obs_buffers[f"{alias}_depth_m"].append(depth.astype(np.float16))
                    obs_buffers[f"{alias}_K"].append(intrinsics)
                    obs_buffers[f"{alias}_T_camera_to_base"].append(extrinsics)
                    obs_buffers[f"{alias}_point_cloud"].append(point_cloud.astype(np.float16))

            ep_grp.create_dataset("actions", data=np.stack(actions).astype(np.float32))
            ep_grp.create_dataset("rlbench_delta_action", data=np.stack(actions).astype(np.float32))
            ep_grp.create_dataset("rlbench_keypose_action", data=np.stack(keypose_actions).astype(np.float32))
            ep_grp.create_dataset("rewards", data=np.zeros(length, dtype=np.float32))
            ep_grp.create_dataset("dones", data=np.zeros(length, dtype=np.uint8))

            obs_grp = ep_grp.create_group("obs")
            obs_grp.create_dataset("proprio", data=np.stack(proprios).astype(np.float32))
            obs_grp.create_dataset("ee_pos", data=np.stack(abs_poses).astype(np.float32)[:, :3])
            obs_grp.create_dataset("rlbench_abs_gripper_pose", data=np.stack(abs_poses).astype(np.float32))
            obs_grp.create_dataset("rlbench_next_abs_gripper_pose", data=np.stack(next_abs_poses).astype(np.float32))
            for key, values in obs_buffers.items():
                obs_grp.create_dataset(key, data=np.stack(values))

            print(f"[ok] task={task} episode={ep_idx} -> demo_{out_idx} transitions={length}")

    print(f"[done] wrote {out_path}")
    return out_path


def export_dataset(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    tasks = parse_csv(args.tasks) if args.tasks else list(DEFAULT_TASKS)
    cameras = tuple(parse_csv(args.cameras))
    if len(cameras) != 2:
        raise ValueError("--cameras must contain exactly two comma-separated RLBench camera names")

    for task in tasks:
        export_task(
            data_root=data_root,
            target_dir=target_dir,
            split=args.split,
            task=task,
            cameras=(cameras[0], cameras[1]),
            max_demos=args.max_demos_per_task,
            overwrite=args.overwrite,
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, help="RLBench dataset root")
    parser.add_argument("--target_dir", required=True, help="Output HDF5 directory")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="Comma-separated task names")
    parser.add_argument("--cameras", default="front,wrist", help="Exactly two RLBench cameras")
    parser.add_argument("--max_demos_per_task", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    export_dataset(build_argparser().parse_args())


if __name__ == "__main__":
    main()
