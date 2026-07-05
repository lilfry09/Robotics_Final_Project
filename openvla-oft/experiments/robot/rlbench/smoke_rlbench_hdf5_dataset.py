"""Create a tiny RLBench-style HDF5 and verify DepthVLA dataset loading."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import torch


def create_synthetic_hdf5(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "rlbench_train_synthetic_task.hdf5"
    steps, height, width = 4, 16, 16
    with h5py.File(path, "w") as f:
        f.attrs["source"] = "rlbench"
        f.attrs["split"] = "train"
        f.attrs["task_name"] = "synthetic_task"
        demo = f.create_group("data/demo_0")
        demo.attrs["language_instruction"] = "insert the peg"

        actions = np.zeros((steps, 7), dtype=np.float32)
        actions[:, 0] = np.linspace(0.01, 0.04, steps)
        actions[:, -1] = 1.0
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("rlbench_delta_action", data=actions)

        keypose = np.zeros((steps, 8), dtype=np.float32)
        keypose[:, :3] = np.stack([np.linspace(0.1, 0.4, steps), np.zeros(steps), np.ones(steps)], axis=1)
        keypose[:, 6] = 1.0
        keypose[:, 7] = 1.0
        demo.create_dataset("rlbench_keypose_action", data=keypose)
        demo.create_dataset("rewards", data=np.zeros(steps, dtype=np.float32))
        demo.create_dataset("dones", data=np.zeros(steps, dtype=np.uint8))

        obs = demo.create_group("obs")
        rgb = np.zeros((steps, height, width, 3), dtype=np.uint8)
        rgb[:, 0, 0, 0] = 255
        wrist = np.zeros((steps, height, width, 3), dtype=np.uint8)
        wrist[:, 0, 0, 1] = 255
        obs.create_dataset("agentview_rgb", data=rgb)
        obs.create_dataset("eye_in_hand_rgb", data=wrist)
        obs.create_dataset("agentview_depth_m", data=np.ones((steps, height, width), dtype=np.float16))
        obs.create_dataset("eye_in_hand_depth_m", data=np.ones((steps, height, width), dtype=np.float16) * 0.5)

        k = np.tile(np.eye(3, dtype=np.float32), (steps, 1, 1))
        k[:, 0, 0] = 20
        k[:, 1, 1] = 20
        k[:, 0, 2] = width / 2
        k[:, 1, 2] = height / 2
        extrinsics = np.tile(np.eye(4, dtype=np.float32), (steps, 1, 1))
        obs.create_dataset("agentview_K", data=k)
        obs.create_dataset("eye_in_hand_K", data=k)
        obs.create_dataset("agentview_T_camera_to_base", data=extrinsics)
        obs.create_dataset("eye_in_hand_T_camera_to_base", data=extrinsics)

        proprio = np.zeros((steps, 8), dtype=np.float32)
        proprio[:, 6] = 1.0
        proprio[:, 7] = 1.0
        obs.create_dataset("proprio", data=proprio)
        obs.create_dataset("ee_pos", data=np.zeros((steps, 3), dtype=np.float32))
        obs.create_dataset("rlbench_abs_gripper_pose", data=keypose)
        obs.create_dataset("rlbench_next_abs_gripper_pose", data=keypose)
    return path


class DummyActionTokenizer:
    def __call__(self, action):
        arr = np.asarray(action).reshape(-1)
        return "".join(["<a>"] * arr.shape[0])


class DummyTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return type("Tok", (), {"input_ids": [1] + [min(ord(c), 200) for c in text[:32]] + [2]})()


class DummyPromptBuilder:
    def __init__(self, *_):
        self.parts = []

    def add_turn(self, role, value):
        self.parts.append(f"{role}:{value}")

    def get_prompt(self):
        return "\n".join(self.parts)


def image_transform(img):
    arr = np.asarray(img).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def load_finetune_module():
    spec = importlib.util.spec_from_file_location("finetune_depthvla", "vla-scripts/finetune_depthvla.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    data_dir = Path("/tmp/depthvla_rlbench_synth")
    create_synthetic_hdf5(data_dir)
    finetune = load_finetune_module()
    dataset = finetune.LiberoRGBDHDF5Dataset(
        data_dir,
        "rlbench_synth",
        DummyActionTokenizer(),
        DummyTokenizer(),
        image_transform,
        DummyPromptBuilder,
        use_depth=True,
        use_proprio=True,
        aux_target="absolute_keypose",
    )
    item = dataset[0]
    uv_dataset = finetune.LiberoRGBDHDF5Dataset(
        data_dir,
        "rlbench_synth",
        DummyActionTokenizer(),
        DummyTokenizer(),
        image_transform,
        DummyPromptBuilder,
        use_depth=True,
        use_proprio=True,
        aux_target="projected_keypose_uv",
    )
    uv_item = uv_dataset[0]
    heatmap_dataset = finetune.LiberoRGBDHDF5Dataset(
        data_dir,
        "rlbench_synth",
        DummyActionTokenizer(),
        DummyTokenizer(),
        image_transform,
        DummyPromptBuilder,
        use_depth=True,
        use_proprio=True,
        aux_target="projected_keypose_heatmap",
        aux_heatmap_size=8,
        aux_heatmap_sigma=1.0,
    )
    heatmap_item = heatmap_dataset[0]
    point_dataset = finetune.LiberoRGBDHDF5Dataset(
        data_dir,
        "rlbench_synth",
        DummyActionTokenizer(),
        DummyTokenizer(),
        image_transform,
        DummyPromptBuilder,
        use_depth=True,
        use_proprio=True,
        aux_target="point_keypose_xyz",
    )
    point_item = point_dataset[0]
    first_close_dataset = finetune.LiberoRGBDHDF5Dataset(
        data_dir,
        "rlbench_synth",
        DummyActionTokenizer(),
        DummyTokenizer(),
        image_transform,
        DummyPromptBuilder,
        use_depth=True,
        use_proprio=True,
        aux_target="first_close_pose_xyz",
    )
    first_close_item = first_close_dataset[0]
    visible_first_close_dataset = finetune.LiberoRGBDHDF5Dataset(
        data_dir,
        "rlbench_synth",
        DummyActionTokenizer(),
        DummyTokenizer(),
        image_transform,
        DummyPromptBuilder,
        use_depth=True,
        use_proprio=True,
        aux_target="visible_first_close_point_xyz",
    )
    visible_first_close_item = visible_first_close_dataset[0]
    pre_first_close_dataset = finetune.LiberoRGBDHDF5Dataset(
        data_dir,
        "rlbench_synth",
        DummyActionTokenizer(),
        DummyTokenizer(),
        image_transform,
        DummyPromptBuilder,
        use_depth=True,
        use_proprio=True,
        aux_target="pre_first_close_pose_xyz",
        aux_future_horizon=2,
    )
    pre_first_close_item = pre_first_close_dataset[0]
    visible_pre_first_close_dataset = finetune.LiberoRGBDHDF5Dataset(
        data_dir,
        "rlbench_synth",
        DummyActionTokenizer(),
        DummyTokenizer(),
        image_transform,
        DummyPromptBuilder,
        use_depth=True,
        use_proprio=True,
        aux_target="visible_pre_first_close_point_xyz",
        aux_future_horizon=2,
    )
    visible_pre_first_close_item = visible_pre_first_close_dataset[0]
    visible_point_dataset = finetune.LiberoRGBDHDF5Dataset(
        data_dir,
        "rlbench_synth",
        DummyActionTokenizer(),
        DummyTokenizer(),
        image_transform,
        DummyPromptBuilder,
        use_depth=True,
        use_proprio=True,
        aux_target="visible_object_point_xyz",
    )
    visible_point_item = visible_point_dataset[0]
    visible_rel_dataset = finetune.LiberoRGBDHDF5Dataset(
        data_dir,
        "rlbench_synth",
        DummyActionTokenizer(),
        DummyTokenizer(),
        image_transform,
        DummyPromptBuilder,
        use_depth=True,
        use_proprio=True,
        aux_target="visible_object_rel_xyz",
    )
    visible_rel_item = visible_rel_dataset[0]
    assert item["actions"].shape == (8, 7)
    assert item["proprio"].shape == (8,)
    assert item["depth_values"].shape == (2, 16, 16)
    assert item["aux_label"].shape == (8,)
    assert uv_item["aux_label"].shape == (4,)
    assert np.isfinite(uv_item["aux_label"]).all()
    assert np.abs(uv_item["aux_label"]).max() <= 2.0
    assert heatmap_item["aux_label"].shape == (2, 8, 8)
    assert np.isfinite(heatmap_item["aux_label"]).all()
    assert float(heatmap_item["aux_label"].max()) > 0.5
    assert point_item["aux_label"].shape == (3,)
    assert np.isfinite(point_item["aux_label"]).all()
    assert first_close_item["aux_label"].shape == (3,)
    assert np.isfinite(first_close_item["aux_label"]).all()
    assert visible_first_close_item["aux_label"].shape == (3,)
    assert np.isfinite(visible_first_close_item["aux_label"]).all()
    assert pre_first_close_item["aux_label"].shape == (3,)
    assert np.isfinite(pre_first_close_item["aux_label"]).all()
    assert visible_pre_first_close_item["aux_label"].shape == (3,)
    assert np.isfinite(visible_pre_first_close_item["aux_label"]).all()
    assert visible_point_item["aux_label"].shape == (3,)
    assert np.isfinite(visible_point_item["aux_label"]).all()
    assert visible_rel_item["aux_label"].shape == (3,)
    assert np.isfinite(visible_rel_item["aux_label"]).all()
    assert abs(float(item["actions"][0, -1]) - 1.0) < 1e-6, "RLBench gripper should not be flipped or zeroed"
    assert float(item["pixel_values"][0, 0, 0]) > 0.9, "RLBench RGB should not be 180-degree rotated"
    print("RLBench HDF5 dataset smoke passed")
    print("  len:", len(dataset))
    print("  action:", tuple(item["actions"].shape))
    print("  proprio:", tuple(item["proprio"].shape))
    print("  depth:", tuple(item["depth_values"].shape))
    print("  aux:", tuple(item["aux_label"].shape))
    print("  projected_uv_aux:", tuple(uv_item["aux_label"].shape), uv_item["aux_label"])
    print("  projected_heatmap_aux:", tuple(heatmap_item["aux_label"].shape), float(heatmap_item["aux_label"].max()))
    print("  point_keypose_xyz_aux:", tuple(point_item["aux_label"].shape), point_item["aux_label"])
    print("  first_close_pose_xyz_aux:", tuple(first_close_item["aux_label"].shape), first_close_item["aux_label"])
    print(
        "  visible_first_close_point_xyz_aux:",
        tuple(visible_first_close_item["aux_label"].shape),
        visible_first_close_item["aux_label"],
    )
    print(
        "  pre_first_close_pose_xyz_aux:",
        tuple(pre_first_close_item["aux_label"].shape),
        pre_first_close_item["aux_label"],
    )
    print(
        "  visible_pre_first_close_point_xyz_aux:",
        tuple(visible_pre_first_close_item["aux_label"].shape),
        visible_pre_first_close_item["aux_label"],
    )
    print("  visible_object_point_xyz_aux:", tuple(visible_point_item["aux_label"].shape), visible_point_item["aux_label"])
    print("  visible_object_rel_xyz_aux:", tuple(visible_rel_item["aux_label"].shape), visible_rel_item["aux_label"])


if __name__ == "__main__":
    main()
