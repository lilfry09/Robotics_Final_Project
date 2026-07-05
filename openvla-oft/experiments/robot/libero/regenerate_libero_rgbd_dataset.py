"""
Regenerates LIBERO demonstrations as RGB-D HDF5 files for DepthVLA-OFT.

This script stores raw, unrotated RGB and metric depth. LIBERO/OpenVLA's
180-degree image rotation is applied by the training/eval dataset code so
geometry can be computed from the original depth pixels and camera matrices.
"""

import argparse
import json
import os
import re

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

VARIATION_SUFFIX_PATTERNS = (
    r"_language_\d+$",
    r"_view_.+$",
    r"_initstate_\d+$",
    r"_table_\d+$",
    r"_tb_\d+$",
    r"_light_\d+$",
    r"_add_\d+$",
    r"_moved_level\d+_sample\d+$",
    r"_level\d+_sample\d+$",
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


def get_inner_libero_env(env):
    """Return the underlying LIBERO task env from an OffScreen/Control wrapper."""
    return getattr(env, "env", env)


def parse_variation_metadata(task_name: str) -> dict:
    metadata = {}
    if "_view_" in task_name:
        prefix, view_suffix = task_name.split("_view_", 1)
        metadata["base_task_name"] = prefix
        view_part = view_suffix
        if "_initstate_" in view_part:
            view_part, init_suffix = view_part.split("_initstate_", 1)
            metadata["initstate"] = init_suffix.split("_")[0]
        view_values = view_part.split("_")
        if len(view_values) >= 5:
            metadata["horizon_view"] = view_values[0]
            metadata["vertical_view"] = view_values[1]
            metadata["scale_percent"] = view_values[2]
            metadata["endpoint_rot"] = view_values[3]
            metadata["endpoint_vertical"] = view_values[4]
    elif "_initstate_" in task_name:
        metadata["base_task_name"] = task_name.split("_initstate_", 1)[0]
        metadata["initstate"] = task_name.split("_initstate_", 1)[1].split("_")[0]
    if "_language_" in task_name:
        metadata["language_variation"] = task_name.split("_language_", 1)[1].split("_")[0]
    if "_add_" in task_name:
        metadata["object_variation"] = "add_" + task_name.split("_add_", 1)[1].split("_")[0]
    if "_level" in task_name:
        metadata["object_variation"] = "level" + task_name.split("_level", 1)[1].split("_")[0]
    return metadata


def parse_csv_strings(value: str, flag_name: str) -> list[str]:
    if not value:
        return []
    parsed = [item.strip().lower() for item in value.split(",") if item.strip()]
    valid = {"object", "camera", "initstate", "language", "light", "table", "base"}
    invalid = sorted(set(parsed) - valid)
    if invalid:
        raise ValueError(f"{flag_name} contains unsupported variation types {invalid}; choose from {sorted(valid)}.")
    return parsed


def classify_variation_types(task_name: str) -> set[str]:
    types = set()
    if "_view_" in task_name:
        types.add("camera")
    if "_initstate_" in task_name:
        types.add("initstate")
    if "_add_" in task_name or "_level" in task_name:
        types.add("object")
    if "_language_" in task_name:
        types.add("language")
    if "_light_" in task_name:
        types.add("light")
    if "_table_" in task_name or "_tb_" in task_name:
        types.add("table")
    if not types:
        types.add("base")
    return types


def parse_csv_ints(value: str, flag_name: str) -> list[int]:
    if not value:
        return []
    parsed = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise ValueError(f"{flag_name} must be a comma-separated list of integers; got {item!r}.") from exc
    return parsed


def infer_base_task_name(task_name: str) -> str:
    base_name = task_name
    changed = True
    while changed:
        changed = False
        for pattern in VARIATION_SUFFIX_PATTERNS:
            stripped = re.sub(pattern, "", base_name)
            if stripped != base_name:
                base_name = stripped
                changed = True
                break
    return base_name


def find_demo_hdf5(raw_data_dir: str, task_name: str) -> str | None:
    filename = f"{task_name}_demo.hdf5"
    direct_path = os.path.join(raw_data_dir, filename)
    if os.path.exists(direct_path):
        return direct_path

    matches = []
    for root, _, files in os.walk(raw_data_dir):
        if filename in files:
            matches.append(os.path.join(root, filename))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileExistsError(f"Found multiple raw demo files for {task_name!r}: {matches}")
    return None


def select_variation_balanced_tasks(
    available_tasks: list[tuple],
    variation_types: list[str],
    max_tasks: int | None,
    max_tasks_per_variation_type: int | None,
    max_variations_per_base_task: int | None,
) -> list[tuple]:
    if not variation_types:
        return available_tasks[:max_tasks] if max_tasks is not None else available_tasks

    selected = []
    selected_task_ids = set()
    type_counts = {variation_type: 0 for variation_type in variation_types}
    base_counts = {}

    def can_select(task_id, task_name, active_type):
        if task_id in selected_task_ids:
            return False
        if max_tasks is not None and len(selected) >= max_tasks:
            return False
        if (
            max_tasks_per_variation_type is not None
            and type_counts.get(active_type, 0) >= max_tasks_per_variation_type
        ):
            return False
        if max_variations_per_base_task is not None:
            base_name = infer_base_task_name(task_name)
            if base_counts.get(base_name, 0) >= max_variations_per_base_task:
                return False
        return True

    made_progress = True
    while made_progress and (max_tasks is None or len(selected) < max_tasks):
        made_progress = False
        for variation_type in variation_types:
            if (
                max_tasks_per_variation_type is not None
                and type_counts.get(variation_type, 0) >= max_tasks_per_variation_type
            ):
                continue
            for candidate in available_tasks:
                task_id, task, _, _ = candidate
                task_types = classify_variation_types(task.name)
                if variation_type not in task_types or not can_select(task_id, task.name, variation_type):
                    continue
                selected.append(candidate)
                selected_task_ids.add(task_id)
                for task_type in task_types:
                    if task_type in type_counts:
                        type_counts[task_type] += 1
                base_name = infer_base_task_name(task.name)
                base_counts[base_name] = base_counts.get(base_name, 0) + 1
                made_progress = True
                break
            if max_tasks is not None and len(selected) >= max_tasks:
                break

    return selected


def get_task_initial_states(task_suite, task_id: int, source: str):
    if source == "raw":
        return None
    if not hasattr(task_suite, "get_task_init_states"):
        raise AttributeError(f"Benchmark {task_suite.name!r} does not expose get_task_init_states().")
    init_states = task_suite.get_task_init_states(task_id)
    init_states = np.asarray(init_states)
    if init_states.ndim == 1:
        init_states = init_states.reshape(1, -1)
    if init_states.shape[0] == 0:
        raise ValueError(f"Benchmark returned no init states for task id {task_id}.")
    return init_states


def select_initial_state(init_states, demo_index: int, orig_states: np.ndarray):
    if init_states is None:
        return orig_states[0], "raw_demo"
    return init_states[demo_index % init_states.shape[0]], "benchmark"


def choose_symbolic_supervision_names(env) -> tuple[str | None, str | None, list]:
    inner_env = get_inner_libero_env(env)
    obj_of_interest = list(getattr(inner_env, "obj_of_interest", []) or [])
    parsed_problem = getattr(inner_env, "parsed_problem", {}) or {}
    goal_state = list(parsed_problem.get("goal_state", []) or [])
    object_names = set(getattr(inner_env, "objects_dict", {}) or {})
    site_names = set(getattr(inner_env, "object_sites_dict", {}) or {})
    fixture_names = set(getattr(inner_env, "fixtures_dict", {}) or {})

    manipulated_object = None
    target_object = None
    for state in goal_state:
        if len(state) < 3:
            continue
        candidate_object = state[1]
        candidate_target = state[2]
        if obj_of_interest and candidate_object not in obj_of_interest:
            continue
        if candidate_object in object_names:
            manipulated_object = candidate_object
            if candidate_target in object_names or candidate_target in site_names or candidate_target in fixture_names:
                target_object = candidate_target
            break

    if manipulated_object is None:
        for name in obj_of_interest:
            if name in object_names:
                manipulated_object = name
                break
    if target_object is None:
        for state in goal_state:
            if len(state) >= 3 and state[2] in object_names | site_names | fixture_names:
                target_object = state[2]
                break
    return manipulated_object, target_object, goal_state


def get_symbolic_pose(env, name: str | None):
    inner_env = get_inner_libero_env(env)
    nan_pos = np.full(3, np.nan, dtype=np.float32)
    nan_quat = np.full(4, np.nan, dtype=np.float32)
    if not name:
        return nan_pos, nan_quat

    object_sites = getattr(inner_env, "object_sites_dict", {}) or {}
    if name in object_sites:
        try:
            site_id = inner_env.sim.model.site_name2id(object_sites[name].name)
            return inner_env.sim.data.site_xpos[site_id].astype(np.float32), nan_quat
        except Exception:
            return nan_pos, nan_quat

    for query_dict in (getattr(inner_env, "objects_dict", {}) or {}, getattr(inner_env, "fixtures_dict", {}) or {}):
        if name not in query_dict:
            continue
        obj = query_dict[name]
        try:
            body_id = inner_env.sim.model.body_name2id(obj.root_body)
            pos = inner_env.sim.data.body_xpos[body_id].astype(np.float32)
            quat = inner_env.sim.data.body_xquat[body_id].astype(np.float32)
            return pos, quat
        except Exception:
            return nan_pos, nan_quat
    return nan_pos, nan_quat


def build_symbolic_geometry(env, ee_pos: np.ndarray, manipulated_object: str | None, target_object: str | None):
    object_pos, object_quat = get_symbolic_pose(env, manipulated_object)
    target_pos, target_quat = get_symbolic_pose(env, target_object)
    ee = np.asarray(ee_pos, dtype=np.float32)[:3]
    ee_to_object = object_pos - ee if np.isfinite(object_pos).all() else np.full(3, np.nan, dtype=np.float32)
    object_to_target = (
        target_pos - object_pos if np.isfinite(object_pos).all() and np.isfinite(target_pos).all() else np.full(3, np.nan, dtype=np.float32)
    )
    distance = np.asarray(
        [np.linalg.norm(ee_to_object) if np.isfinite(ee_to_object).all() else np.nan], dtype=np.float32
    )
    return {
        "manipulated_object_pos": object_pos,
        "manipulated_object_quat": object_quat,
        "target_pos": target_pos,
        "target_quat": target_quat,
        "ee_to_object_xyz": ee_to_object.astype(np.float32),
        "object_to_target_xyz": object_to_target.astype(np.float32),
        "gripper_to_contact_distance": distance,
    }


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
    if args.max_tasks_per_variation_type is not None and args.max_tasks_per_variation_type <= 0:
        raise ValueError("--max_tasks_per_variation_type must be positive when set.")
    if args.max_variations_per_base_task is not None and args.max_variations_per_base_task <= 0:
        raise ValueError("--max_variations_per_base_task must be positive when set.")

    requested_task_ids = set(parse_csv_ints(args.task_ids, "--task_ids"))
    requested_variation_types = parse_csv_strings(args.variation_types, "--variation_types")
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
    invalid_task_ids = sorted(task_id for task_id in requested_task_ids if task_id < 0 or task_id >= num_tasks_in_suite)
    if invalid_task_ids:
        raise ValueError(f"Requested task ids out of range for {args.libero_task_suite}: {invalid_task_ids}")

    available_tasks = []
    for task_id in range(num_tasks_in_suite):
        if requested_task_ids and task_id not in requested_task_ids:
            continue
        task = task_suite.get_task(task_id)
        if requested_task_names and task.name not in requested_task_names:
            continue
        if requested_variation_types and not (classify_variation_types(task.name) & set(requested_variation_types)):
            continue
        raw_task_name = infer_base_task_name(task.name) if args.use_base_demos_for_variations else task.name
        orig_data_path = find_demo_hdf5(args.libero_raw_data_dir, raw_task_name)
        if orig_data_path is None:
            message = (
                f"Cannot find raw demo for selected task '{task.name}' "
                f"(looked for raw task '{raw_task_name}') under {args.libero_raw_data_dir}."
            )
            if args.skip_missing:
                print(f"Skipping missing task '{task.name}' using raw demo task '{raw_task_name}'. {message}")
                continue
            raise FileNotFoundError(message)
        available_tasks.append((task_id, task, raw_task_name, orig_data_path))

    selected_tasks = select_variation_balanced_tasks(
        available_tasks,
        requested_variation_types,
        args.max_tasks,
        args.max_tasks_per_variation_type,
        args.max_variations_per_base_task,
    )

    if requested_task_names:
        selected_task_names = {task.name for _, task, _, _ in selected_tasks}
        missing_requested = sorted(requested_task_names - selected_task_names)
        if missing_requested and not args.skip_missing:
            raise FileNotFoundError(f"Requested task names not found in raw data: {missing_requested}")
    if requested_task_ids:
        selected_task_ids = {task_id for task_id, _, _, _ in selected_tasks}
        missing_requested = sorted(requested_task_ids - selected_task_ids)
        if missing_requested and not args.skip_missing:
            raise FileNotFoundError(f"Requested task ids not found in raw data: {missing_requested}")

    if len(selected_tasks) == 0:
        raise FileNotFoundError("No tasks selected for RGB-D regeneration.")

    print("Selected tasks:")
    for task_id, task, raw_task_name, orig_data_path in selected_tasks:
        raw_note = f" (raw demo: {raw_task_name})" if raw_task_name != task.name else ""
        variation_note = ",".join(sorted(classify_variation_types(task.name)))
        print(f"  - [{task_id}] [{variation_note}] {task.name}{raw_note} -> {orig_data_path}")

    num_replays = 0
    num_success = 0
    num_noops = 0

    for task_id, task, raw_task_name, orig_data_path in tqdm.tqdm(selected_tasks):
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION, camera_depths=True)
        manipulated_object, target_object, goal_state = choose_symbolic_supervision_names(env)
        variation_metadata = parse_variation_metadata(task.name)
        variation_metadata["base_task_name"] = infer_base_task_name(task.name)
        variation_metadata["raw_demo_task_name"] = raw_task_name
        variation_metadata["raw_demo_path"] = orig_data_path
        print(f"  symbolic object={manipulated_object}, target={target_object}")

        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]
        benchmark_init_states = get_task_initial_states(task_suite, task_id, args.initial_state_source)

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
            init_state, init_state_source = select_initial_state(benchmark_init_states, i, orig_states)

            env.reset()
            obs = env.set_init_state(init_state)
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
            symbolic_buffers = {
                "manipulated_object_pos": [],
                "manipulated_object_quat": [],
                "target_pos": [],
                "target_quat": [],
                "ee_to_object_xyz": [],
                "object_to_target_xyz": [],
                "gripper_to_contact_distance": [],
            }

            for _, action in enumerate(orig_actions):
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    print(f"\tSkipping no-op action: {action}")
                    num_noops += 1
                    continue

                if states == []:
                    states.append(np.asarray(init_state))
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )
                else:
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                actions.append(action)
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_state = np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])))
                ee_states.append(ee_state)
                symbolic_geometry = build_symbolic_geometry(
                    env, ee_state[:3], manipulated_object=manipulated_object, target_object=target_object
                )
                for key, value in symbolic_geometry.items():
                    symbolic_buffers[key].append(value)

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
                ep_data_grp.attrs["task_name"] = task.name
                ep_data_grp.attrs["raw_demo_task_name"] = raw_task_name
                ep_data_grp.attrs["initial_state_source"] = init_state_source
                ep_data_grp.attrs["manipulated_object_name"] = manipulated_object or ""
                ep_data_grp.attrs["target_object_name"] = target_object or ""
                ep_data_grp.attrs["goal_state_json"] = json.dumps(goal_state)
                ep_data_grp.attrs["variation_metadata_json"] = json.dumps(variation_metadata)
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                for key, values in camera_buffers.items():
                    obs_grp.create_dataset(key, data=np.stack(values, axis=0))
                for key, values in symbolic_buffers.items():
                    obs_grp.create_dataset(key, data=np.stack(values, axis=0).astype(np.float32))
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
            metainfo_json_dict[task_key][episode_key]["task_id"] = task_id
            metainfo_json_dict[task_key][episode_key]["task_name"] = task.name
            metainfo_json_dict[task_key][episode_key]["raw_demo_task_name"] = raw_task_name
            metainfo_json_dict[task_key][episode_key]["initial_state_source"] = init_state_source
            metainfo_json_dict[task_key][episode_key]["initial_state"] = np.asarray(init_state).tolist()
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
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
            "libero_mix",
            "libero_plus",
            "libero_pro",
        ],
        required=True,
    )
    parser.add_argument("--libero_raw_data_dir", type=str, required=True)
    parser.add_argument("--libero_target_dir", type=str, required=True)
    parser.add_argument("--task_ids", type=str, default="", help="Comma-separated benchmark task ids to regenerate.")
    parser.add_argument("--task_names", type=str, default="", help="Comma-separated task names to regenerate.")
    parser.add_argument(
        "--variation_types",
        type=str,
        default="",
        help=(
            "Comma-separated variation types to include and balance across: "
            "object,camera,initstate,language,light,table,base."
        ),
    )
    parser.add_argument(
        "--use_base_demos_for_variations",
        action="store_true",
        help="Replay actions from the matching base task demo when the selected benchmark task is a LIBERO-Plus variation.",
    )
    parser.add_argument(
        "--initial_state_source",
        type=str,
        choices=["raw", "benchmark"],
        default="raw",
        help="Use raw demo initial states or benchmark initial states for each selected task.",
    )
    parser.add_argument("--max_tasks", type=int, default=None, help="Maximum number of available tasks to regenerate.")
    parser.add_argument(
        "--max_tasks_per_variation_type",
        type=int,
        default=None,
        help="Maximum selected tasks per requested variation type when --variation_types is set.",
    )
    parser.add_argument(
        "--max_variations_per_base_task",
        type=int,
        default=None,
        help="Maximum selected variations sharing the same inferred base task when --variation_types is set.",
    )
    parser.add_argument("--max_demos_per_task", type=int, default=None, help="Maximum demos to replay per task.")
    parser.add_argument("--skip_missing", action="store_true", help="Skip tasks whose raw HDF5 file is missing.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip target HDF5 files that already exist.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files directly inside the target directory.")
    main(parser.parse_args())
