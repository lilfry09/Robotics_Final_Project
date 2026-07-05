# RLBench RGB-D Adapter

This folder is the new starting point for moving DepthVLA-OFT away from saturated clean LIBERO evaluation.

Goal:

```text
RLBench demos
  -> DepthVLA-compatible HDF5
  -> matched RGB-only / RGB-D training
  -> normal/null/shuffle depth rollout ablations
```

Why RLBench:

- It contains manipulation tasks with stronger 3D geometry requirements than clean LIBERO.
- PerAct/RVT/Act3D/BridgeVLA-style methods already use RLBench to evaluate 3D action grounding.
- RLBench demonstrations expose RGB, depth, point cloud, camera intrinsics/extrinsics, gripper pose, and language descriptions.

First target:

```text
6 tasks x 10 demos:
  slide_block_to_target
  turn_tap
  close_jar
  open_drawer
  reach_target
  pick_up_cup
```

The converter writes the existing two-view DepthVLA fields:

- `agentview_rgb`
- `eye_in_hand_rgb`
- `agentview_depth_m`
- `eye_in_hand_depth_m`
- `agentview_K`
- `eye_in_hand_K`
- `agentview_T_camera_to_base`
- `eye_in_hand_T_camera_to_base`
- `actions`
- `proprio`

It also writes new labels for the next round of experiments:

- `rlbench_abs_gripper_pose`
- `rlbench_next_abs_gripper_pose`
- `rlbench_delta_action`
- `rlbench_keypose_action`

Model-side support already added:

- `prismatic/models/dense_point_depth_encoder.py`
- `vla-scripts/finetune_depthvla.py --depth_encoder_type dense_point`
- `vla-scripts/finetune_depthvla.py --aux_target absolute_keypose --aux_output_dim 8`

Probe/gate scripts:

- `validate_rlbench_hdf5.py`
- `smoke_rlbench_hdf5_dataset.py`
- `probe_dense_depth_keypose.py`
- `compare_rgbd_rollout_results.py`

Stage runner:

- `run_rlbench_rgbd_stage.sh`
- `rlbench_stage.env.example`
- `ensure_xvfb.sh`

Dry-run:

```bash
DRY_RUN=1 experiments/robot/rlbench/run_rlbench_rgbd_stage.sh dry-run
```

Environment status after setup:

```text
rlbench / pyrep / peract_colab / yarr: ok
CoppeliaSim: /root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
PyRep CFFI: ok
Xvfb display: :1.0
```

Generate the first real pilot demos:

```bash
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh xvfb
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh generate-demos
```

Then run the data gates:

```bash
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh convert
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh validate
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh probe
```

Final causal rollout gate:

```bash
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh gate-results
```

By default this expects four result files:

```text
experiments/logs/rlbench_eval_results/rgb_only.json
experiments/logs/rlbench_eval_results/rgbd_normal.json
experiments/logs/rlbench_eval_results/rgbd_null.json
experiments/logs/rlbench_eval_results/rgbd_shuffle.json
```

The gate passes only if:

- RGB-D normal beats matched RGB-only by at least `0.05` success rate.
- RGB-D normal beats null depth by at least `0.05`.
- RGB-D normal beats shuffled depth by at least `0.05`.

Important:

This is an adapter scaffold. It expects RLBench/PerAct-style demos to be available locally and RLBench/PyRep/peract helper packages to be installed.
