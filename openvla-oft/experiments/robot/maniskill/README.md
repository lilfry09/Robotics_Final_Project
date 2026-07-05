# ManiSkill3 RGB-D / Point-Cloud Adapter Plan

更新时间：2026-07-04 UTC

## 目标

当前 RLBench/OpenVLA-OFT residual 和 waypoint recipe 已经多轮 no-go。下一轮正结果路线不应继续扩大同一 recipe，而应先接 ManiSkill3 这种高吞吐 RGB-D / point-cloud 环境，并训练一个 primary 3D action decoder。

当前最重要的本地结果：

```text
learned-phase object-feature policy:
  normal 19/30, null 0/30, cross_demo 0/30

raw cropped pointcloud policy, 30 teacher episodes:
  normal 2/30, null 0/30, cross_demo 0/30

raw cropped pointcloud policy, 100 teacher episodes, two eval seeds:
  normal 20/60, null 1/60, cross_demo 1/60

matched no-depth train baselines, same data/model/eval seeds:
  sampled-RGB-only 1/60, null/proprio 3/60

learned raw-pointcloud cube predictor + fixed geometry controller:
  normal 22/30, null 1/30, cross_demo 0/30
```

这些是 ManiSkill teacher-distillation / diagnostic 结果，不是 OpenVLA 端到端结果；但它们已经说明 PickCube 是更合适的 3D-sensitive benchmark，raw pointcloud 可以被 learned policy 转成闭环收益。

第一阶段目标不是直接 claim RGB-D improvement，而是跑通：

```text
ManiSkill3 task
  -> RGB-D / pointcloud / proprio / action / language
  -> unified HDF5
  -> normal/null/cross-sample action-delta diagnostic
  -> matched RGB-only vs RGB-D gate
```

## 当前本机状态

为了不污染 `depthvla`，ManiSkill3 已安装到独立 venv：

```text
/root/autodl-tmp/envs/maniskill3-venv
```

`depthvla` conda 环境仍不安装 ManiSkill3；运行 ManiSkill adapter 时使用：

```bash
export PYTHON_BIN=/root/autodl-tmp/envs/maniskill3-venv/bin/python
export MS_ASSET_DIR=/root/autodl-tmp/maniskill_data
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
```

环境检查：

```bash
/root/autodl-tmp/envs/maniskill3-venv/bin/python experiments/robot/maniskill/check_maniskill_env.py
```

已通过：

```text
mani_skill: ok (3.0.1)
gymnasium: ok (1.3.0)
torch: ok (2.12.1+cu130)
```

SAPIEN 会报告 system Vulkan library / ICD warning，但 `PushCube-v1` 的 state 和 pointcloud smoke 均已成功。

## 最小 HDF5 Schema

建议先导出一个 task 的小样本，字段与 RLBench HDF5 尽量对齐：

```text
observations/rgb/<camera>          uint8 [T,H,W,3]
observations/depth/<camera>        float32 [T,H,W]
observations/pointcloud/xyz        float32 [T,N,3]
observations/proprio               float32 [T,P]
actions                            float32 [T,H,A] or [T,A]
language                           string
episode_id                         int
task_name                          string
success                            bool or int
camera_intrinsics/<camera>         float32 [3,3]
camera_extrinsics/<camera>         float32 [4,4]
```

如果第一步只训练 DP3-style decoder，可以先要求 `pointcloud/xyz`、`proprio`、`actions` 和 `language`，RGB 可以稍后接回 OpenVLA。

## 第一批 Gate

不要一开始做大表。先做这三个检查：

1. **Data smoke**：一个 task 至少导出 `10` 条 demo，所有 point/action/proprio finite。
2. **Offline action loss**：RGB-D normal action loss 低于 null/cross-sample。
3. **Paired action delta**：同一 observation 下 normal vs cross-sample depth/pointcloud 产生可观 action 差异，不能再是 `1e-4` 级弱扰动。

只有这三个 gate 通过，才进入 rollout 或扩大训练。

## 已完成 Smoke

State observation：

```bash
MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/collect_random_hdf5.py \
  --env_id PushCube-v1 \
  --obs_mode state \
  --episodes 1 \
  --max_steps 2 \
  --output experiments/logs/maniskill_pushcube_state_smoke.hdf5
```

Pointcloud observation：

```bash
MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/collect_random_hdf5.py \
  --env_id PushCube-v1 \
  --obs_mode pointcloud \
  --episodes 1 \
  --max_steps 2 \
  --output experiments/logs/maniskill_pushcube_pointcloud_smoke.hdf5
```

Validation：

```bash
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/validate_maniskill_hdf5.py \
  experiments/logs/maniskill_pushcube_pointcloud_smoke.hdf5
```

Pointcloud smoke 里已经包含：

```text
obs/pointcloud/xyzw: [T, 16384, 4]
obs/pointcloud/rgb: [T, 16384, 3]
obs/pointcloud/segmentation: [T, 16384, 1]
obs/sensor_param/base_camera/intrinsic_cv: [T, 3, 3]
obs/sensor_param/base_camera/extrinsic_cv: [T, 3, 4]
obs/agent/qpos, obs/agent/qvel, actions
```

## 推荐模型路线

优先级：

1. DP3-style：sparse point cloud encoder + proprio/language embedding + action chunk decoder。
2. Act3D/PerAct-style：workspace 3D action map / point action classification。
3. OpenVLA-OFT fusion：只作为后续重新接 VLA 的路线，不再先做 residual patch。

## 最后 pilot：official demo replay + point-cloud decoder

提交前补做了一个小规模 offline action gate。它不是 rollout success，也不是最终 RGB-D improvement claim，但比 RLBench/OpenVLA residual/waypoint recipe 更能产生可测的 pointcloud-action coupling。

### Replay 官方 demo 为 pointcloud

官方 `replay_trajectory` 支持把 demo 重新保存成目标 observation/control mode。注意：`--use-env-states` 不能和 `--target-control-mode` conversion 同时使用，因此这里选择转换到 `pd_ee_delta_pos`，不用 `--use-env-states`。

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

### 摘要和验证

```bash
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/summarize_demo_hdf5.py \
  /root/autodl-tmp/maniskill_data/PickCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pos.physx_cpu.h5

/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/validate_maniskill_hdf5.py \
  /root/autodl-tmp/maniskill_data/PickCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pos.physx_cpu.h5
```

### 训练最小 point-cloud action decoder

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

### 当前结果

| task | demos | pointcloud gate | paired normal-vs-cross L2 mean |
|---|---:|---:|---:|
| `PushCube-v1` | `20` | `3/3` seeds | `0.002041` |
| `PickCube-v1` | `20` | `2/3` seeds | `0.022263` |

解读：

- PushCube 是平面推方块，适合 smoke，但 pointcloud 相对 proprio-only 的优势很小。
- PickCube 更 3D-sensitive，normal-vs-cross action delta 明显更大。
- 这仍然是 offline action prediction，不是 rollout；下一步必须接 closed-loop rollout 和 matched no-depth/RGB baseline。

### Scaled raw pointcloud teacher policy

最后一轮把 geometry-teacher pointcloud 数据从 30 条成功轨迹扩到 100 条，并训练 h256/10k 单步 PointNet action decoder：

```bash
MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/collect_geometry_teacher_pointcloud_dataset.py \
  --output experiments/logs/maniskill_pickcube_geometry_teacher_pointcloud_cropz002_cubeaux_success100_seed6000.npz \
  --summary experiments/logs/maniskill_pickcube_geometry_teacher_pointcloud_cropz002_cubeaux_success100_seed6000.json \
  --target_successes 100 \
  --max_attempts 180 \
  --max_steps 150 \
  --num_points 512 \
  --min_z 0.02 \
  --seed 6000

MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/train_pointcloud_teacher_phase_action_decoder.py \
  --input experiments/logs/maniskill_pickcube_geometry_teacher_pointcloud_cropz002_cubeaux_success100_seed6000.npz \
  --output experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_seed7_10k_h256.json \
  --checkpoint_output experiments/logs/maniskill_checkpoints/pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_seed7_10k_h256.pt \
  --steps 10000 \
  --batch_size 256 \
  --eval_batch_size 512 \
  --hidden_dim 256 \
  --lr 3e-4 \
  --phase_loss_weight 0.2 \
  --cube_loss_weight 0.5 \
  --seed 7
```

闭环评估：

```bash
MS_ASSET_DIR=/root/autodl-tmp/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/eval_pointcloud_teacher_phase_action_decoder.py \
  --checkpoint experiments/logs/maniskill_checkpoints/pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_seed7_10k_h256.pt \
  --output experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_normal_seed7_150steps_30eps.json \
  --point_mode normal \
  --episodes 30 \
  --max_steps 150 \
  --min_z 0.02 \
  --seed 4100
```

把 `--point_mode` 改成 `null` 或 `cross_demo`，cross-demo 还需要：

```bash
--cross_dataset experiments/logs/maniskill_pickcube_geometry_teacher_pointcloud_cropz002_cubeaux_success100_seed6000.npz
```

结果：

| eval seed | normal | null | cross_demo |
|---|---:|---:|---:|
| `4100` | `8/30` | `1/30` | `1/30` |
| `4500` | `12/30` | `0/30` | `0/30` |
| aggregate | `20/60` | `1/60` | `1/60` |

Matched no-depth training baselines:

| train input | rollout input | aggregate success |
|---|---|---:|
| normal pointcloud xyz/rgb | normal | `20/60` |
| sampled RGB only, xyz zeroed | rgb_only | `1/60` |
| null points + task state/proprio | null | `3/60` |

同一数据的 h=8 action-chunk decoder 离线 gate 通过，但闭环没有超过单步模型；因此下一步应换 ACT/DP3/diffusion/recurrent policy，而不是继续堆这个浅 chunk decoder。

### Closed-loop smoke

已补一个最小闭环 smoke：

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

结果：

| checkpoint | mode | success | mean reward |
|---|---|---:|---:|
| pointcloud 1200-step | normal | `0/3` | `7.09` |
| pointcloud 1200-step | null | `0/3` | `4.32` |
| pointcloud 1200-step | cross_demo | `0/3` | `7.33` |
| proprio 1200-step | null | `0/3` | `6.81` |
| pointcloud 5000-step | normal | `0/3` | `10.31` |
| pointcloud 5000-step | null | `0/3` | `2.91` |
| pointcloud 5000-step | cross_demo | `0/3` | `10.75` |

结论：offline action coupling 没有直接转成 closed-loop success。下一步需要 action chunk / diffusion decoder / 更强 3D action map，而不是单步 PointNet BC。

### Action chunk smoke

最小 action chunk 版本：

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

结果：

| mode | execute steps | success | mean reward |
|---|---:|---:|---:|
| normal | `4` | `0/3` | `5.46` |
| null | `4` | `0/3` | `3.01` |
| cross_demo | `4` | `0/3` | `5.46` |
| normal | `1` | `0/3` | `3.57` |

### Geometry-teacher distillation

最后补了一个更强的诊断：用 pointcloud geometry controller 生成成功 teacher rollouts，再训练 object-feature MLP。

无 phase 版本：

| mode | success | mean reward |
|---|---:|---:|
| normal | `0/10` | `54.56` |
| null | `0/10` | `3.67` |
| cross_demo | `0/10` | `12.83` |

它能抓住 cube，但不会稳定进入 move-goal 阶段。

加入 phase one-hot 后，action 仍由 learned MLP 输出，phase 由几何状态机提供：

| eval | normal | null | cross_demo |
|---|---:|---:|---:|
| 10 episodes | `7/10` | `0/10` | `0/10` |
| 30 episodes | `17/30` | `0/30` | `0/30` |

Phase/geometry 解耦对照，相同 seed，10 episodes：

| action geometry | phase source | success | mean reward |
|---|---|---:|---:|
| normal | normal | `6/10` | `39.23` |
| null | normal | `0/10` | `4.47` |
| cross_demo | normal | `0/10` | `14.28` |
| normal | null | `0/10` | `14.06` |

结论：这是 learned positive diagnostic，不是端到端 OpenVLA 结果。它说明真实 pointcloud geometry 能驱动 learned action decoder 超过 null/cross controls；下一步要用 ACT/DP3/recurrent/diffusion policy 学到 phase/temporal state，而不是手写 phase。

结论：简单 action chunk 离线可学，但没有产生 reliable closed-loop success。下一步应改成 DP3/diffusion/ACT-style temporal policy。

### Geometry controller diagnostic

显式几何控制器：

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

结果：

| mode | success | mean reward |
|---|---:|---:|
| normal pointcloud | `7/10` | `26.38` |
| null pointcloud | `0/10` | `5.84` |
| cross_demo pointcloud | `0/10` | `9.23` |

结论：PickCube 是 pointcloud geometry-solvable 的；当前 learned decoder 没成功，是因为没有学到稳定几何控制/temporal policy。

### Raw cropped pointcloud + learned cube controller

最终又补了一个更接近 raw RGB-D/pointcloud 的诊断。输入不包含 `cube_center`，只使用 `z>0.02` cropped pointcloud + task state；训练时用 cube center auxiliary supervision。

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

Offline gate:

| mode | action RMSE | cube RMSE |
|---|---:|---:|
| normal | `0.103` | `0.009m` |
| null | `0.184` | `0.076m` |
| cross_sample | `0.215` | `0.075m` |

Closed-loop:

| policy | normal | null | cross_demo |
|---|---:|---:|---:|
| learned action head, 30 teacher episodes | `2/30` | `0/30` | `0/30` |
| learned action head, 100 teacher episodes, aggregate | `20/60` | `1/60` | `1/60` |
| matched sampled-RGB-only train baseline | `1/60` | - | - |
| matched null/proprio train baseline | `3/60` | - | - |
| learned cube + fixed controller | `22/30` | `1/30` | `0/30` |

结论：raw pointcloud perception 已经足够强；扩大 teacher 数据后 learned action 也能明显超过 null/cross，但仍远低于 hybrid controller。下一步应把 object-centric bottleneck 接入 ACT/DP3/diffusion action policy。
