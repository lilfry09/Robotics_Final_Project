"""
Regenerates LIBERO demonstrations as RGB-D HDF5 files for DepthVLA-OFT.

This script stores raw, unrotated RGB and metric depth. LIBERO/OpenVLA's
180-degree image rotation is applied by the training/eval dataset code so
geometry can be computed from the original depth pixels and camera matrices.
"""

import argparse
import json
import os

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark
from robosuite.utils import camera_utils

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.regenerate_libero_dataset import is_noop


IMAGE_RESOLUTION = 256
CAMERA_SPECS = (
    ("agentview", "agentview_image", "agentview_depth"),
    ("robot0_eye_in_hand", "robot0_eye_in_hand_image", "robot0_eye_in_hand_depth"),
)


def get_camera_rgbd_and_geometry(env, obs, camera_name: str, rgb_key: str, depth_key: str):
    if depth_key not in obs:
        raise KeyError(
            f"Observation is missing `{depth_key}`. Make sure the LIBERO environment was created with camera_depths=True."
        )
    raw_depth = obs[depth_key]
    depth_m = camera_utils.get_real_depth_map(env.sim, raw_depth).astype(np.float32)
    intrinsics = camera_utils.get_camera_intrinsic_matrix(
        env.sim,
        camera_name=camera_name,
        camera_height=IMAGE_RESOLUTION,
        camera_width=IMAGE_RESOLUTION,
    ).astype(np.float32)
    extrinsics = camera_utils.get_camera_extrinsic_matrix(env.sim, camera_name=camera_name).astype(np.float32)
    return obs[rgb_key], depth_m, intrinsics, extrinsics


def main(args):
    print(f"Regenerating RGB-D {args.libero_task_suite} dataset!")
    if os.path.isdir(args.libero_target_dir):
        if args.overwrite:
            print(f"Overwriting existing target directory: {args.libero_target_dir}")
            for filename in os.listdir(args.libero_target_dir):
                path = os.path.join(args.libero_target_dir, filename)
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    raise IsADirectoryError(f"Refusing to remove nested directory: {path}")
        elif not args.skip_existing:
            user_input = input(
                f"Target directory already exists at path: {args.libero_target_dir}\n"
                "Enter 'y' to overwrite the directory, or anything else to exit: "
            )
            if user_input != "y":
                return
    elif args.skip_existing:
        print("--skip_existing was set, but target directory does not exist yet.")
    os.makedirs(args.libero_target_dir, exist_ok=True)

    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("--max_tasks must be positive when set.")
    if args.max_demos_per_task is not None and args.max_demos_per_task <= 0:
        raise ValueError("--max_demos_per_task must be positive when set.")

    requested_task_names = set()
    if args.task_names:
        requested_task_names = {task_name.strip() for task_name in args.task_names.split(",") if task_name.strip()}
        if len(requested_task_names) == 0:
            raise ValueError("--task_names was provided but no non-empty task names were parsed.")

    if args.skip_existing:
        print("Will skip target files that already exist.")

    metainfo_json_dict = {}
    metainfo_json_out_path = f"./experiments/robot/libero/{args.libero_task_suite}_rgbd_metainfo.json"
    with open(metainfo_json_out_path, "w") as f:
        json.dump(metainfo_json_dict, f)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    selected_tasks = []
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        if requested_task_names and task.name not in requested_task_names:
            continue
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        if not os.path.exists(orig_data_path):
            message = f"Cannot find raw data file {orig_data_path}."
            if args.skip_missing:
                print(f"Skipping missing task '{task.name}'. {message}")
                continue
            raise FileNotFoundError(message)
        selected_tasks.append((task_id, task))
        if args.max_tasks is not None and len(selected_tasks) >= args.max_tasks:
            break

    if requested_task_names:
        selected_task_names = {task.name for _, task in selected_tasks}
        missing_requested = sorted(requested_task_names - selected_task_names)
        if missing_requested and not args.skip_missing:
            raise FileNotFoundError(f"Requested task names not found in raw data: {missing_requested}")

    if len(selected_tasks) == 0:
        raise FileNotFoundError("No tasks selected for RGB-D regeneration.")

    print("Selected tasks:")
    for _, task in selected_tasks:
        print(f"  - {task.name}")

    num_replays = 0
    num_success = 0
    num_noops = 0

    for task_id, task in tqdm.tqdm(selected_tasks):
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION, camera_depths=True)

        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        if args.skip_existing and os.path.exists(new_data_path):
            print(f"Skipping existing regenerated file: {new_data_path}")
            orig_data_file.close()
            continue
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        demo_keys = sorted(orig_data.keys(), key=lambda key: int(key.split("_")[-1]))
        if args.max_demos_per_task is not None:
            demo_keys = demo_keys[: args.max_demos_per_task]

        for demo_key in demo_keys:
            i = int(demo_key.split("_")[-1])
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            states, actions, ee_states = [], [], []
            gripper_states, joint_states, robot_states = [], [], []
            camera_buffers = {
                "agentview_rgb": [],
                "eye_in_hand_rgb": [],
                "agentview_depth_m": [],
                "eye_in_hand_depth_m": [],
                "agentview_K": [],
                "eye_in_hand_K": [],
                "agentview_T_camera_to_base": [],
                "eye_in_hand_T_camera_to_base": [],
            }

            for _, action in enumerate(orig_actions):
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    print(f"\tSkipping no-op action: {action}")
                    num_noops += 1
                    continue

                if states == []:
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                actions.append(action)
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]))))

                agent_rgb, agent_depth, agent_k, agent_t = get_camera_rgbd_and_geometry(
                    env, obs, "agentview", "agentview_image", "agentview_depth"
                )
                wrist_rgb, wrist_depth, wrist_k, wrist_t = get_camera_rgbd_and_geometry(
                    env, obs, "robot0_eye_in_hand", "robot0_eye_in_hand_image", "robot0_eye_in_hand_depth"
                )
                camera_buffers["agentview_rgb"].append(agent_rgb)
                camera_buffers["eye_in_hand_rgb"].append(wrist_rgb)
                camera_buffers["agentview_depth_m"].append(agent_depth.astype(np.float16))
                camera_buffers["eye_in_hand_depth_m"].append(wrist_depth.astype(np.float16))
                camera_buffers["agentview_K"].append(agent_k)
                camera_buffers["eye_in_hand_K"].append(wrist_k)
                camera_buffers["agentview_T_camera_to_base"].append(agent_t)
                camera_buffers["eye_in_hand_T_camera_to_base"].append(wrist_t)

                obs, reward, done, info = env.step(action.tolist())

            if done:
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1

                ep_data_grp = grp.create_group(f"demo_{i}")
                ep_data_grp.attrs["language_instruction"] = task_description
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                for key, values in camera_buffers.items():
                    obs_grp.create_dataset(key, data=np.stack(values, axis=0))
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)
                num_success += 1

            num_replays += 1
            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"
            metainfo_json_dict.setdefault(task_key, {}).setdefault(episode_key, {})
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: "
                f"{num_success} ({num_success / num_replays * 100:.1f} %)"
            )
            print(f"  Total # no-op actions filtered out: {num_noops}")

        orig_data_file.close()
        new_data_file.close()
        print(f"Saved regenerated RGB-D demos for task '{task_description}' at: {new_data_path}")

    print(f"RGB-D dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90", "libero_plus", "libero_pro"],
        required=True,
    )
    parser.add_argument("--libero_raw_data_dir", type=str, required=True)
    parser.add_argument("--libero_target_dir", type=str, required=True)
    parser.add_argument("--task_names", type=str, default="", help="Comma-separated task names to regenerate.")
    parser.add_argument("--max_tasks", type=int, default=None, help="Maximum number of available tasks to regenerate.")
    parser.add_argument("--max_demos_per_task", type=int, default=None, help="Maximum demos to replay per task.")
    parser.add_argument("--skip_missing", action="store_true", help="Skip tasks whose raw HDF5 file is missing.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip target HDF5 files that already exist.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files directly inside the target directory.")
    main(parser.parse_args())
