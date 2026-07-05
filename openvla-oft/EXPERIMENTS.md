# DepthVLA-OFT Experiments

更新时间：2026-07-04 UTC

## 1. 当前最终状态

当前最终可复现结论是：

> 在更 3D-sensitive 的 ManiSkill3 PickCube 上，raw pointcloud learned policy 已经能把 depth/pointcloud 几何转成闭环收益：normal `20/60`，null `1/60`，cross-demo `1/60`。

提交前的最后 ManiSkill3 结果给出了更强的正向诊断，但必须限定范围：

| result | normal | null | cross_demo | interpretation |
|---|---:|---:|---:|---|
| learned-phase object-feature policy | `19/30` | `0/30` | `0/30` | learned policy can use true object geometry, but input is segmentation-derived features |
| raw cropped pointcloud policy, 30 teacher eps | `2/30` | `0/30` | `0/30` | weak raw-pointcloud learned rollout signal |
| raw cropped pointcloud policy, 100 teacher eps, aggregate | `20/60` | `1/60` | `1/60` | strongest raw-pointcloud learned action result |
| matched sampled-RGB-only train baseline | `1/60` | - | - | same data/model/eval seeds, no xyz geometry |
| matched null/proprio train baseline | `3/60` | - | - | same data/model/eval seeds, no point input |
| learned cube + fixed geometry controller | `22/30` | `1/30` | `0/30` | perception is strong; action/temporal decoding remains the bottleneck |

因此可以说“ManiSkill3/PickCube 上 raw pointcloud teacher-distilled policy 已经出现 normal > null/cross 的闭环收益”。OpenVLA/LIBERO 结果作为背景解释：原来的 benchmark 和 optional adapter 不适合证明 depth value。

最终结果表：

```bash
python scripts/collect_depthvla_final_results.py
```

输出：

```text
FINAL_RESULTS_TABLE.md
experiments/logs/final_results_table.csv
```

## 2. 最后一次正式训练：open_drawer primary waypoint-action

训练命令：

```bash
HDF5_DIR=/root/RLBench/rgbd_hdf5_open_drawer_3demos_64 \
DATASET_NAME=rlbench_open_drawer_3demos_64 \
RUN_ROOT_DIR=/root/runs_rlbench_open_drawer_waypoint_action \
TASKS=open_drawer \
MAX_STEPS=5000 \
SAVE_FREQ=1000 \
DEPTH_POINTS_PER_VIEW=1024 \
DEPTH_ACTION_FUSION_GATE_INIT=1.0 \
DEPTH_HIDDEN_DELTA_CLIP=0.001 \
DEPTH_ACTION_RESIDUAL_CLIP=0.0 \
DEPTH_AUX_TARGET=point_keypose_xyz \
DEPTH_AUX_OUTPUT_DIM=3 \
DEPTH_WAYPOINT_ACTION_WEIGHT=1.0 \
DEPTH_WAYPOINT_ACTION_CLIP=0.02 \
DEPTH_AUX_SPATIAL_LOSS_WEIGHT=1.0 \
DEPTH_DROPOUT=0.0 \
FREEZE_VLA_LORA=True \
FREEZE_PROPRIO_PROJECTOR=True \
FREEZE_ACTION_HEAD_BASE=True \
RESUME_COMPONENTS_FROM=/root/runs_rlbench_open_drawer_3demos/47a0ec7fc4ec123775a391911046cf33cf9ed83f+rlbench_open_drawer_3demos_64+rgb-only+b1+lr-0.0001+lora-r4+dropout-0.0--rlbench-rgb-only \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgbd
```

训练完成后先跑 paired causal diagnostic：

```bash
RGBD_CHECKPOINT=/root/runs_rlbench_open_drawer_waypoint_action/47a0ec7fc4ec123775a391911046cf33cf9ed83f+rlbench_open_drawer_3demos_64+depth-densep1024+object-query+gate-1.0+wpact-1.0+wpclip-0.02+aux-point_keypose_xyz-1.0+b1+lr-0.0001+lora-r4+dropout-0.0--rlbench-rgbd-dense-keypose \
HDF5_DIR=/root/RLBench/rgbd_hdf5_open_drawer_3demos_64 \
DATASET_NAME=rlbench_open_drawer_3demos_64 \
TASKS=open_drawer \
LOG_DIR=/root/autodl-tmp/openvla-oft/experiments/logs/rlbench_open_drawer_waypoint_action \
DEPTH_AUX_OUTPUT_DIM=3 \
DEPTH_WAYPOINT_ACTION_WEIGHT=1.0 \
DEPTH_WAYPOINT_ACTION_CLIP=0.02 \
DEPTH_HIDDEN_DELTA_CLIP=0.001 \
DEPTH_ACTION_RESIDUAL_CLIP=0.0 \
DIAG_COMPARE_DEPTH_MODE=cross_sample \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh diagnose-rgbd-normal
```

再跑 strict diagnostic：

```bash
RGBD_CHECKPOINT=<same checkpoint> \
HDF5_DIR=/root/RLBench/rgbd_hdf5_open_drawer_3demos_64 \
DATASET_NAME=rlbench_open_drawer_3demos_64 \
TASKS=open_drawer \
DEPTH_AUX_OUTPUT_DIM=3 \
DEPTH_WAYPOINT_ACTION_WEIGHT=1.0 \
DEPTH_WAYPOINT_ACTION_CLIP=0.02 \
DEPTH_HIDDEN_DELTA_CLIP=0.001 \
DEPTH_ACTION_RESIDUAL_CLIP=0.0 \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh diagnose-rgbd-all-strict
```

最后只做小规模 rollout sanity：

```bash
RGBD_CHECKPOINT=<same checkpoint> \
HDF5_DIR=/root/RLBench/rgbd_hdf5_open_drawer_3demos_64 \
DATASET_NAME=rlbench_open_drawer_3demos_64 \
TASKS=open_drawer \
EVAL_DATA_ROOT=/root/RLBench/peract_dataset/stable6_3demos_64 \
EVAL_RESULT_DIR=/root/autodl-tmp/openvla-oft/experiments/logs/rlbench_open_drawer_waypoint_action/eval_h200 \
EVAL_EPISODES=1 \
EVAL_EPISODE_LENGTH=200 \
EVAL_IMAGE_SIZE=64 \
DEPTH_POINTS_PER_VIEW=1024 \
DEPTH_AUX_OUTPUT_DIM=3 \
DEPTH_WAYPOINT_ACTION_WEIGHT=1.0 \
DEPTH_WAYPOINT_ACTION_CLIP=0.02 \
DEPTH_HIDDEN_DELTA_CLIP=0.001 \
DEPTH_ACTION_RESIDUAL_CLIP=0.0 \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh eval-rgbd-all-strict
```

实际结果：NO-GO。详见 `FINAL_RESULTS_TABLE.md`。

## 3. 3D action-map feasibility probe

为了判断下一轮是否值得做 Act3D/PerAct-style action map，新增了候选点覆盖率 probe：

```bash
/root/miniconda3/envs/depthvla/bin/python experiments/robot/rlbench/probe_3d_action_map_feasibility.py \
  --data_dir /root/RLBench/rgbd_hdf5_open_drawer_3demos_64 \
  --target keypose \
  --points_per_view 1024 \
  --stride 1 \
  --output experiments/logs/rlbench_3d_action_map_feasibility_open_drawer.json

/root/miniconda3/envs/depthvla/bin/python experiments/robot/rlbench/probe_3d_action_map_feasibility.py \
  --data_dir /root/RLBench/rgbd_hdf5_stable6_3demos_64 \
  --target keypose \
  --points_per_view 1024 \
  --stride 2 \
  --output experiments/logs/rlbench_3d_action_map_feasibility_stable6_stride2.json
```

关键结果：

| setting | target | normal median | cross median | EE fallback median |
|---|---|---:|---:|---:|
| `open_drawer` | keypose/next | `0.0376m` | `0.0724m` | `0.0091m` |
| `open_drawer` | future10 | `0.0426m` | `0.0598m` | `0.0800m` |
| `open_drawer` | farthest future | `0.0466m` | `0.0738m` | `0.1982m` |
| stable6 stride2 | keypose/next | `0.0376m` | `0.1039m` | `0.0081m` |
| stable6 stride2 | future10 | `0.0411m` | `0.1061m` | `0.0777m` |
| stable6 stride2 | final | `0.0203m` | `0.0781m` | `0.2331m` |
| stable6 stride2 | farthest future | `0.0191m` | `0.0792m` | `0.2587m` |

解读：

> 短步 keypose/next-pose target 太接近当前末端执行器，仍然容易被 proprio shortcut 解决；future/final/farthest-future target 则让 normal point candidates 同时优于 cross-sample 和 EE fallback，更适合下一轮 3D action-map 监督。

## 4. 不再扩大的原因

当前 causal gate 没过：

- paired normal-vs-cross action delta 只有 `3.05e-04` 级别。
- strict normal/null/cross `xyz_rmse` 基本打平。
- rollout normal/null/cross 全部 `0/1`，并触发 `InvalidActionError`。

因此不应该继续用同一 recipe 扩大训练量。扩大训练量的前置条件是：

```text
normal depth 明显优于 null/cross-sample
normal depth 优于 matched RGB-only
提升出现在 3D/contact/viewpoint-sensitive tasks
```

此外，3D action-map feasibility probe 显示当前 label 设计本身也有 shortcut 风险：如果 target 只是短步 next pose，action map 也不一定能强制使用 depth。下一轮应改成 future/final/farthest-future 或 object/contact-conditioned target。

## 5. 最后一轮可复现改进：long-horizon 3D target

为了避免 `point_keypose_xyz` / next-pose target 太接近当前末端执行器，训练入口已经支持更长视野的 3D auxiliary target：

- `future_pose_xyz`：当前 step 后第 `DEPTH_AUX_FUTURE_HORIZON` 帧的绝对 EE xyz。
- `final_pose_xyz`：episode 最后一帧 EE xyz。
- `farthest_future_pose_xyz`：从当前时刻往后、距离当前 EE 最远的未来 EE xyz。

这些 target 的目的不是立刻 claim 正结果，而是给下一轮 primary waypoint / 3D action-map 训练去掉短步 proprio shortcut。

训练命令模板：

```bash
HDF5_DIR=/root/RLBench/rgbd_hdf5_open_drawer_3demos_64 \
DATASET_NAME=rlbench_open_drawer_3demos_64 \
RUN_ROOT_DIR=/root/runs_rlbench_open_drawer_farthest_future \
TASKS=open_drawer \
MAX_STEPS=5000 \
SAVE_FREQ=1000 \
DEPTH_POINTS_PER_VIEW=1024 \
DEPTH_ACTION_FUSION_GATE_INIT=1.0 \
DEPTH_HIDDEN_DELTA_CLIP=0.001 \
DEPTH_ACTION_RESIDUAL_CLIP=0.0 \
DEPTH_AUX_TARGET=farthest_future_pose_xyz \
DEPTH_AUX_OUTPUT_DIM=3 \
DEPTH_AUX_FUTURE_HORIZON=10 \
DEPTH_WAYPOINT_ACTION_WEIGHT=1.0 \
DEPTH_WAYPOINT_ACTION_CLIP=0.02 \
DEPTH_AUX_SPATIAL_LOSS_WEIGHT=1.0 \
DEPTH_DROPOUT=0.0 \
FREEZE_VLA_LORA=True \
FREEZE_PROPRIO_PROJECTOR=True \
FREEZE_ACTION_HEAD_BASE=True \
RESUME_COMPONENTS_FROM=/root/runs_rlbench_open_drawer_3demos/47a0ec7fc4ec123775a391911046cf33cf9ed83f+rlbench_open_drawer_3demos_64+rgb-only+b1+lr-0.0001+lora-r4+dropout-0.0--rlbench-rgb-only \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgbd
```

已验证：

```text
future_pose_xyz / final_pose_xyz / farthest_future_pose_xyz 能在真实 open_drawer HDF5 上生成 finite `(3,)` 标签。
runner dry-run 能正确传入 `--aux_future_horizon 10`。
`MAX_STEPS=1` 真实训练 smoke 通过：aux target 为 `farthest_future_pose_xyz`，prediction/label shape 均为 `(1, 3)`。
```

500-step 门槛实验：

```bash
RUN_ROOT_DIR=/root/autodl-tmp/openvla-oft/runs_rlbench_farthest_future_small \
HDF5_DIR=/root/RLBench/rgbd_hdf5_open_drawer_3demos_64 \
DATASET_NAME=rlbench_open_drawer_3demos_64 \
TASKS=open_drawer \
MAX_STEPS=500 \
SAVE_FREQ=500 \
DEPTH_POINTS_PER_VIEW=256 \
DEPTH_AUX_TARGET=farthest_future_pose_xyz \
DEPTH_AUX_OUTPUT_DIM=3 \
DEPTH_AUX_FUTURE_HORIZON=10 \
DEPTH_WAYPOINT_ACTION_WEIGHT=1.0 \
DEPTH_WAYPOINT_ACTION_CLIP=0.02 \
DEPTH_AUX_SPATIAL_LOSS_WEIGHT=1.0 \
FREEZE_VLA_LORA=True \
FREEZE_PROPRIO_PROJECTOR=True \
FREEZE_ACTION_HEAD_BASE=True \
RESUME_COMPONENTS_FROM=/root/runs_rlbench_open_drawer_3demos/47a0ec7fc4ec123775a391911046cf33cf9ed83f+rlbench_open_drawer_3demos_64+rgb-only+b1+lr-0.0001+lora-r4+dropout-0.0--rlbench-rgb-only \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgbd
```

诊断结果：

| check | value |
|---|---:|
| paired normal-vs-cross `paired_pred_l1` | `2.65e-05` |
| paired normal-vs-cross `paired_pred_rmse` | `6.26e-05` |
| paired normal-vs-cross `paired_pred_xyz_l2` | `1.66e-04` |
| strict normal `xyz_rmse` | `0.003190` |
| strict null `xyz_rmse` | `0.003210` |
| strict cross-sample `xyz_rmse` | `0.003189` |

判定：NO-GO。`farthest_future_pose_xyz` 的 500-step 小实验没有增强 depth causal effect，paired action delta 反而小于上一轮 primary waypoint-action。

该 no-go checkpoint 已删除以节省磁盘；保留证据为上述 JSON 日志、结果表和复现命令。

5000-step 最后确认实验：

```bash
RUN_ROOT_DIR=/root/autodl-tmp/openvla-oft/runs_rlbench_farthest_future_5k \
HDF5_DIR=/root/RLBench/rgbd_hdf5_open_drawer_3demos_64 \
DATASET_NAME=rlbench_open_drawer_3demos_64 \
TASKS=open_drawer \
MAX_STEPS=5000 \
SAVE_FREQ=5000 \
BATCH_SIZE=1 \
DEPTH_POINTS_PER_VIEW=256 \
DEPTH_ACTION_FUSION_GATE_INIT=1.0 \
DEPTH_HIDDEN_DELTA_CLIP=0.001 \
DEPTH_ACTION_RESIDUAL_CLIP=0.0 \
DEPTH_AUX_TARGET=farthest_future_pose_xyz \
DEPTH_AUX_OUTPUT_DIM=3 \
DEPTH_AUX_FUTURE_HORIZON=10 \
DEPTH_WAYPOINT_ACTION_WEIGHT=1.0 \
DEPTH_WAYPOINT_ACTION_CLIP=0.02 \
DEPTH_AUX_SPATIAL_LOSS_WEIGHT=1.0 \
DEPTH_DROPOUT=0.0 \
FREEZE_VLA_LORA=True \
FREEZE_PROPRIO_PROJECTOR=True \
FREEZE_ACTION_HEAD_BASE=True \
RESUME_COMPONENTS_FROM=/root/runs_rlbench_open_drawer_3demos/47a0ec7fc4ec123775a391911046cf33cf9ed83f+rlbench_open_drawer_3demos_64+rgb-only+b1+lr-0.0001+lora-r4+dropout-0.0--rlbench-rgb-only \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgbd
```

诊断结果：

| check | value |
|---|---:|
| paired normal-vs-cross `paired_pred_l1` | `1.58e-05` |
| paired normal-vs-cross `paired_pred_rmse` | `3.79e-05` |
| paired normal-vs-cross `paired_pred_xyz_l2` | `1.00e-04` |
| strict normal `xyz_rmse` | `0.003167` |
| strict null `xyz_rmse` | `0.003210` |
| strict cross-sample `xyz_rmse` | `0.003162` |

判定：NO-GO。训练到 `5000` steps 后，paired depth effect 更小，strict diagnostic 中 cross-sample 还略优于 normal。因此不跑 rollout，不继续扩大同一 waypoint recipe。

该 no-go checkpoint 已删除以节省磁盘；保留证据为：

```text
experiments/logs/rlbench_farthest_future_5k/
experiments/logs/rlbench_farthest_future_5k_strict/
```

## 6. ManiSkill3 最后 pilot

最后补做了一个高吞吐数据/动作空间方向的 offline pilot。它不是 rollout 结果，但用于判断下一条路线是否比 RLBench/OpenVLA-OFT residual recipe 更有希望。

生成 PickCube pointcloud demo：

```bash
MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  -m mani_skill.trajectory.replay_trajectory \
  --traj-path /root/autodl-tmp/maniskill_data/PickCube-v1/motionplanning/trajectory.h5 \
  --obs-mode pointcloud \
  --target-control-mode pd_ee_delta_pos \
  --save-traj \
  --count 20 \
  --allow-failure \
  --max-retry 0
```

训练 point-cloud decoder gate：

```bash
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/train_pointcloud_action_decoder.py \
  --input /root/autodl-tmp/maniskill_data/PickCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pos.physx_cpu.h5 \
  --output experiments/logs/maniskill_pickcube_pointcloud_decoder_gate_20demo_strictcross_seed7.json \
  --num_points 512 \
  --steps 1200 \
  --batch_size 64 \
  --eval_batch_size 128 \
  --hidden_dim 128 \
  --lr 3e-4 \
  --min_paired_delta 1e-3 \
  --seed 7
```

结果摘要：

| task | demos | pointcloud gate | normal RMSE mean | null RMSE mean | cross RMSE mean | paired normal-vs-cross L2 mean |
|---|---:|---:|---:|---:|---:|---:|
| `PushCube-v1` | `20` | `3/3` seeds | `0.015634` | `0.020435` | `0.015687` | `0.002041` |
| `PickCube-v1` | `20` | `2/3` seeds | `0.119985` | `0.140440` | `0.120763` | `0.022263` |

判定：promising but preliminary。PickCube 的 action sensitivity 比 RLBench farthest-future waypoint 的 `1e-4` 明显更强，但它仍然只是 offline action prediction；下一步必须跑 closed-loop 和 matched no-depth/RGB/proprio baseline。

闭环 smoke：

```bash
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/eval_pointcloud_action_decoder.py \
  --checkpoint experiments/logs/maniskill_checkpoints/pickcube_pointcloud_seed7_5k.pt \
  --output experiments/logs/maniskill_pickcube_rollout_pointcloud_normal_seed7_5k.json \
  --env_id PickCube-v1 \
  --control_mode pd_ee_delta_pos \
  --point_mode normal \
  --episodes 3 \
  --max_steps 100 \
  --seed 4100
```

| checkpoint | mode | success | mean reward |
|---|---|---:|---:|
| pointcloud 1200-step | normal | `0/3` | `7.09` |
| pointcloud 1200-step | null | `0/3` | `4.32` |
| pointcloud 1200-step | cross_demo | `0/3` | `7.33` |
| proprio 1200-step | null | `0/3` | `6.81` |
| pointcloud 5000-step | normal | `0/3` | `10.31` |
| pointcloud 5000-step | null | `0/3` | `2.91` |
| pointcloud 5000-step | cross_demo | `0/3` | `10.75` |

判定：closed-loop 仍是 NO-GO。离线 gate 能放大，但 tiny single-step BC decoder 没有完成 PickCube；下一步应使用 action chunk / diffusion action decoder / 更完整的 3D action-map policy。

Action chunk 尝试：

```bash
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/train_pointcloud_action_chunk_decoder.py \
  --input /root/autodl-tmp/maniskill_data/PickCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pos.physx_cpu.h5 \
  --output experiments/logs/maniskill_pickcube_pointcloud_chunk_gate_h8_seed7_5k.json \
  --checkpoint_output experiments/logs/maniskill_checkpoints/pickcube_pointcloud_chunk_h8_seed7_5k.pt \
  --num_points 512 \
  --chunk_horizon 8 \
  --steps 5000 \
  --batch_size 64 \
  --eval_batch_size 128 \
  --hidden_dim 128 \
  --lr 3e-4 \
  --seed 7
```

| mode | execute steps | success | mean reward |
|---|---:|---:|---:|
| normal | `4` | `0/3` | `5.46` |
| null | `4` | `0/3` | `3.01` |
| cross_demo | `4` | `0/3` | `5.46` |
| normal | `1` | `0/3` | `3.57` |

判定：action chunk 离线 gate 通过，但 closed-loop 仍 NO-GO。normal 高于 null，但没有超过 cross_demo，因此不能 claim 真实几何闭环收益。

Goal-conditioned PointNet 尝试：

```bash
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/train_pointcloud_goal_action_decoder.py \
  --input /root/autodl-tmp/maniskill_data/PickCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pos.physx_cpu.h5 \
  --output experiments/logs/maniskill_pickcube_goal_pointcloud_gate_seed7_5k.json \
  --checkpoint_output experiments/logs/maniskill_checkpoints/pickcube_goal_pointcloud_seed7_5k.pt \
  --num_points 512 \
  --steps 5000 \
  --batch_size 64 \
  --eval_batch_size 128 \
  --hidden_dim 128 \
  --lr 3e-4 \
  --seed 7
```

| mode | offline RMSE | success | mean reward |
|---|---:|---:|---:|
| normal | `0.196182` | `0/3` | `5.97` |
| null | `0.206774` | `0/3` | `5.53` |
| cross_sample / cross_demo | `0.195966` | `0/3` | `6.61` |

判定：NO-GO。加入 `goal_pos` 后 normal 优于 null，但没有超过 cross_sample，闭环仍失败。

Object-centric feature MLP 尝试：

```bash
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/train_object_feature_action_decoder.py \
  --input /root/autodl-tmp/maniskill_data/PickCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pos.physx_cpu.h5 \
  --output experiments/logs/maniskill_pickcube_object_feature_gate_seed7_5k.json \
  --checkpoint_output experiments/logs/maniskill_checkpoints/pickcube_object_feature_seed7_5k.pt \
  --steps 5000 \
  --batch_size 128 \
  --eval_batch_size 256 \
  --hidden_dim 128 \
  --lr 3e-4 \
  --seed 7
```

闭环评测：

```bash
MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/eval_object_feature_action_decoder.py \
  --checkpoint experiments/logs/maniskill_checkpoints/pickcube_object_feature_seed7_5k.pt \
  --output experiments/logs/maniskill_pickcube_rollout_object_feature_normal_seed7_5k_100steps_10eps.json \
  --point_mode normal \
  --episodes 10 \
  --max_steps 100 \
  --seed 4100
```

| mode | offline RMSE | success | mean reward |
|---|---:|---:|---:|
| normal | `0.130584` | `0/10` | `4.83` |
| null | `51700.929688` | `0/10` | `0.51` |
| cross_sample / cross_demo | `0.990587` | `0/10` | `5.25` |

| metric | value |
|---|---:|
| paired normal-vs-cross L2 | `1.314387` |
| offline gate | passed |

注意：null RMSE 极大主要来自 `cube_valid=0` 的 OOD 输入；这组证据应优先看 normal vs cross_sample。

判定：离线 GO，闭环 NO-GO。object-centric 3D features 已经让 action prediction 强依赖真实几何，但单步 BC 没有学到稳定时序控制。

Geometry-teacher distillation：

```bash
MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/collect_geometry_teacher_dataset.py \
  --output experiments/logs/maniskill_pickcube_geometry_teacher_success30_phase_seed5300.npz \
  --summary experiments/logs/maniskill_pickcube_geometry_teacher_success30_phase_seed5300.json \
  --target_successes 30 \
  --max_attempts 60 \
  --max_steps 150 \
  --seed 5300 \
  --include_phase
```

```bash
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/train_teacher_object_feature_decoder.py \
  --input experiments/logs/maniskill_pickcube_geometry_teacher_success30_phase_seed5300.npz \
  --output experiments/logs/maniskill_pickcube_teacher_object_feature_phase_gate_success30_seed7_5k.json \
  --checkpoint_output experiments/logs/maniskill_checkpoints/pickcube_teacher_object_feature_phase_success30_seed7_5k.pt \
  --steps 5000 \
  --batch_size 256 \
  --eval_batch_size 512 \
  --hidden_dim 128 \
  --lr 3e-4 \
  --seed 7
```

```bash
MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/eval_object_feature_action_decoder.py \
  --checkpoint experiments/logs/maniskill_checkpoints/pickcube_teacher_object_feature_phase_success30_seed7_5k.pt \
  --output experiments/logs/maniskill_pickcube_rollout_teacher_object_feature_phase_normal_success30_seed7_5k_150steps_30eps.json \
  --point_mode normal \
  --episodes 30 \
  --max_steps 150 \
  --seed 6100
```

结果：

| eval | normal | null | cross_demo |
|---|---:|---:|---:|
| 10 episodes | `7/10` | `0/10` | `0/10` |
| 30 episodes | `17/30` | `0/30` | `0/30` |

30-episode mean reward：

| mode | mean reward |
|---|---:|
| normal | `38.28` |
| null | `4.15` |
| cross_demo | `17.42` |

Phase/geometry 解耦对照：

```bash
MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/eval_object_feature_action_decoder.py \
  --checkpoint experiments/logs/maniskill_checkpoints/pickcube_teacher_object_feature_phase_success30_seed7_5k.pt \
  --output experiments/logs/maniskill_pickcube_rollout_teacher_object_feature_phase_nullgeom_normalphase_seed7_150steps_10eps.json \
  --point_mode null \
  --phase_source normal \
  --episodes 10 \
  --max_steps 150 \
  --seed 7100
```

| action geometry | phase source | success | mean reward |
|---|---|---:|---:|
| normal | normal | `6/10` | `39.23` |
| null | normal | `0/10` | `4.47` |
| cross_demo | normal | `0/10` | `14.28` |
| normal | null | `0/10` | `14.06` |

判定：learned positive diagnostic，但不是完整 OpenVLA/RGB-D result。它使用手写 phase 状态机提供 phase one-hot，action 由 learned MLP 输出；说明下一步最应该让 ACT/DP3/recurrent/diffusion policy 自己学习 temporal state。

Geometry controller positive diagnostic：

```bash
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/eval_pointcloud_geometry_controller.py \
  --output experiments/logs/maniskill_pickcube_geometry_normal_seed4100_goalz_10eps.json \
  --point_mode normal \
  --episodes 10 \
  --max_steps 100 \
  --seed 4100 \
  --gain 5.0 \
  --max_xyz_action 0.25
```

| mode | success | mean reward |
|---|---:|---:|
| normal pointcloud | `7/10` | `26.38` |
| null pointcloud | `0/10` | `5.84` |
| cross_demo pointcloud | `0/10` | `9.23` |

150-step / last-cube-memory 版本：

| mode | success | mean reward |
|---|---:|---:|
| normal pointcloud | `8/10` | `34.70` |
| null pointcloud | `0/10` | `8.76` |
| cross_demo pointcloud | `1/10` | `17.66` |

判定：positive diagnostic。这个几何 controller 不是 learned policy，但它证明 PickCube 可以由当前 pointcloud geometry 闭环解决，并且真实 pointcloud 内容对成功是必要的。learned decoder 的失败点已经收紧到 temporal policy / compounding error。

Raw cropped pointcloud + learned cube diagnostic：

```bash
MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/collect_geometry_teacher_pointcloud_dataset.py \
  --output experiments/logs/maniskill_pickcube_geometry_teacher_pointcloud_cropz002_cubeaux_success30_seed5600.npz \
  --summary experiments/logs/maniskill_pickcube_geometry_teacher_pointcloud_cropz002_cubeaux_success30_seed5600.json \
  --target_successes 30 \
  --max_attempts 60 \
  --max_steps 150 \
  --num_points 512 \
  --min_z 0.02 \
  --seed 5600
```

```bash
MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/train_pointcloud_teacher_phase_action_decoder.py \
  --input experiments/logs/maniskill_pickcube_geometry_teacher_pointcloud_cropz002_cubeaux_success30_seed5600.npz \
  --output experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success30_seed7_5k.json \
  --checkpoint_output experiments/logs/maniskill_checkpoints/pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success30_seed7_5k.pt \
  --steps 5000 \
  --batch_size 128 \
  --eval_batch_size 256 \
  --hidden_dim 128 \
  --lr 3e-4 \
  --phase_loss_weight 0.2 \
  --cube_loss_weight 0.5 \
  --seed 7
```

Offline gate：

| mode | action RMSE | phase acc | cube RMSE |
|---|---:|---:|---:|
| normal | `0.103` | `94.3%` | `0.009m` |
| null | `0.184` | `89.5%` | `0.076m` |
| cross_sample | `0.215` | `87.7%` | `0.075m` |

同一个 checkpoint 的 learned action head 闭环：

```bash
MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/eval_pointcloud_teacher_phase_action_decoder.py \
  --checkpoint experiments/logs/maniskill_checkpoints/pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success30_seed7_5k.pt \
  --output experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_normal_success30_seed7_5k_150steps_30eps.json \
  --point_mode normal \
  --episodes 30 \
  --max_steps 150 \
  --min_z 0.02 \
  --seed 4100
```

| policy | normal | null | cross_demo |
|---|---:|---:|---:|
| learned action head, 30 teacher episodes | `2/30` | `0/30` | `0/30` |
| learned action head, 100 teacher episodes, aggregate | `20/60` | `1/60` | `1/60` |
| matched sampled-RGB-only train baseline | `1/60` | - | - |
| matched null/proprio train baseline | `3/60` | - | - |

把同一个 learned cube predictor 接到固定 geometry controller：

```bash
MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/eval_pointcloud_cube_controller.py \
  --checkpoint experiments/logs/maniskill_checkpoints/pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success30_seed7_5k.pt \
  --output experiments/logs/maniskill_pickcube_rollout_learned_cube_controller_cropz002_normal_seed7_150steps_30eps.json \
  --point_mode normal \
  --episodes 30 \
  --max_steps 150 \
  --min_z 0.02 \
  --seed 4100
```

| hybrid diagnostic | normal | null | cross_demo |
|---|---:|---:|---:|
| learned cube + fixed controller | `22/30` | `1/30` | `0/30` |

判定：raw pointcloud perception 已经足够强；扩大 teacher 数据后 learned action 明显超过 eval-time null/cross 和 train-time no-depth baselines，但 hybrid controller 仍更强，说明 temporal/action decoder 还可以继续改。这个结果是 ManiSkill RGB-D/pointcloud policy diagnostic，不是 OpenVLA 端到端结果。

详细记录见 `MANISKILL_FINAL_PILOT.md`。

## 7. 下一轮正结果路线

下一轮不要继续小 residual patch，优先做完整 action-space redesign：

1. RLBench 或 ManiSkill3 数据。
2. object/contact-conditioned 3D target，而不是短步 next-pose label。
3. 3D action map / voxel keypose / coarse-to-fine action detection。
4. 或 DP3-style point-cloud diffusion action decoder。
5. 在 `open_drawer` 或同类 3D-sensitive task 上先过 normal/null/cross causal gate。
6. 过 gate 后再扩到 stable6 `10 demos/task` 或 ManiSkill3 大规模 synthetic RGB-D。

最小 go/no-go：

```text
GO:
  paired normal-vs-cross selected action target clearly nonzero
  normal strict diagnostic better than null/cross
  normal rollout > null/cross and > matched RGB-only

NO-GO:
  normal/null/cross strict metrics nearly tied
  rollout all fail or cross/null also succeeds
  selected spatial target has signal but final action remains unchanged
```
