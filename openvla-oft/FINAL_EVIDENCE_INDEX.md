# DepthVLA-OFT 最终证据索引

更新时间：2026-07-05 UTC

## 1. 最终主张

最终最有用的结论是：

> 在更 3D-sensitive 的 ManiSkill3 PickCube 上，真实 depth/pointcloud 几何可以被 learned policy 因果使用，并且明显超过 eval-time corrupt controls 和 train-time no-depth baselines。

最核心的证据表：

| setting | success | mean reward |
|---|---:|---:|
| learned raw pointcloud policy, normal input | `20/60` | `32.42` |
| same policy, eval-time null input | `1/60` | `16.78` |
| same policy, eval-time cross-demo input | `1/60` | `17.34` |
| matched sampled-RGB-only train baseline | `1/60` | `16.69` |
| matched null/proprio train baseline | `3/60` | `14.54` |
| learned cube + fixed controller, normal/null/cross | `22/30` / `1/30` / `0/30` | `27.86` / `18.50` / `10.28` |

边界要讲清楚，但不要讲成负结论：

- 当前最强正证据来自 ManiSkill3 PickCube teacher-distilled PointNet/pointcloud policy，而不是 OpenVLA 端到端设置。
- 这不证明 OpenVLA-OFT RGB-D 没用；只能说明我们这轮 LIBERO/RLBench residual/waypoint recipe 没有给出 matched positive evidence。
- 因此答辩时主 claim 放在 “depth/pointcloud 几何在合适 benchmark 和 primary action decoder 下可以转成闭环收益”，OpenVLA 部分作为方法边界和 no-go 对照。

## 2. 核心日志

Raw pointcloud policy gate：

```text
experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_seed7_10k_h256.json
experiments/logs/maniskill_checkpoints/pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_seed7_10k_h256.pt
```

Normal pointcloud rollout：

```text
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_normal_seed7_150steps_30eps.json
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_normal_seed7_150steps_seed4500_30eps.json
```

Eval-time corrupt controls：

```text
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_null_seed7_150steps_30eps.json
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_null_seed7_150steps_seed4500_30eps.json
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_cross_demo_seed7_150steps_30eps.json
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_cross_demo_seed7_150steps_seed4500_30eps.json
```

Matched train-time no-depth baselines:

```text
experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_rgbonly_seed7_10k_h256.json
experiments/logs/maniskill_checkpoints/pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_rgbonly_seed7_10k_h256.pt
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_rgbonly_seed7_150steps_30eps.json
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_rgbonly_seed7_150steps_seed4500_30eps.json
experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_nulltrain_seed7_10k_h256.json
experiments/logs/maniskill_checkpoints/pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_nulltrain_seed7_10k_h256.pt
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_nulltrain_seed7_150steps_30eps.json
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_nulltrain_seed7_150steps_seed4500_30eps.json
```

## 3. 为什么 baseline 是 matched

这些 baseline 与 normal pointcloud policy 保持一致：

- 同一个 ManiSkill3 `PickCube-v1` task。
- 同一套 `100` 条成功 geometry-teacher rollouts，`8388` transitions。
- 同样 `z>0.02` point sampling 入口和 task state/proprio 输入。
- 同样 h256 PointNet teacher action decoder。
- 同样 `10000` train steps。
- 同样两个 30-episode eval seeds：`4100` 和 `4500`。

差别只在 point input：

| baseline | point input |
|---|---|
| normal pointcloud | xyz/rgb 都保留 |
| sampled-RGB-only | xyz 置零，只保留 sampled RGB |
| null/proprio | point input 全零，只保留 task state/proprio |

因此 `20/60` vs `1/60` / `3/60` 可以作为 “pointcloud geometry > no-depth baseline” 的主证据。

## 4. 离线 gate

Raw pointcloud h256/10k gate：

| mode | action RMSE | phase acc | cube RMSE |
|---|---:|---:|---:|
| normal | `0.067` | `98.0%` | `0.0079m` |
| null | `0.210` | `95.0%` | `0.0547m` |
| cross_sample | `0.131` | `97.0%` | `0.0590m` |

Paired normal-vs-cross action L2：`0.120`。

注意：`rgb_only` / `null_train` baseline 的离线 BC loss 也可以很低，但闭环 success 低。这正好说明只看 offline action loss 会误导，真正关键是 closed-loop + matched controls。

## 5. 其他正向诊断

Learned-phase object-feature policy：

| setting | normal | null | cross_demo |
|---|---:|---:|---:|
| learned phase, 30 episodes | `19/30` | `0/30` | `0/30` |

Learned-phase disentanglement：

| action geometry | learned phase source | success |
|---|---|---:|
| null | normal | `0/10` |
| cross_demo | normal | `0/10` |
| normal | null | `0/10` |

Raw pointcloud perception/action split：

| setting | normal | null | cross_demo |
|---|---:|---:|---:|
| learned action head, 30 teacher eps | `2/30` | `0/30` | `0/30` |
| learned action head, 100 teacher eps | `20/60` | `1/60` | `1/60` |
| learned cube + fixed controller | `22/30` | `1/30` | `0/30` |

解读：

> raw pointcloud perception 已经够强；扩大 teacher 数据后 learned action 也明显吃到几何信息。剩余提升空间主要在 temporal/action decoder。

## 6. OpenVLA/RLBench 的最终边界

OpenVLA-OFT / RLBench residual 和旧 waypoint 路线没有形成 rollout 正结果。最后一轮 `visible_pre_first_close_point_xyz + 8-step waypoint action chunk` 给出了更清楚的几何正诊断：normal depth 让 selected 3D point 更接近当前可见 pre-contact label，并且这个点进入 temporal action chunk。

- clean LIBERO/RGB-only 接近天花板，不适合证明 depth 边际收益。
- RLBench projected heatmap probe 证明 depth 有空间信号。
- 旧 OpenVLA-OFT optional residual / unscaled waypoint action path 没有把 depth 稳定转成 closed-loop gain。
- 早期 visible-object geometry bottleneck 证明 depth 可以进入 OpenVLA 的 selected 3D point、waypoint action，并最终改变反归一化 action。
- 最新 visible-precontact gate 进一步证明 selected-point geometry 本身通过 normal/null/cross-sample gate，但 strict action imitation 仍未过 cross-sample gate。

旧 no-go：

```text
experiments/logs/rlbench_open_drawer_waypoint_action/rlbench_policy_action_diag_rgbd_normal.json
experiments/logs/rlbench_open_drawer_waypoint_action/eval_h200/rgbd_normal.json
experiments/logs/rlbench_farthest_future_5k/
experiments/logs/rlbench_farthest_future_5k_strict/
```

过程 OpenVLA 正向诊断：

```text
runs_rlbench_visible_object_point_scale20_500/
experiments/logs/visible_object_point_scale20_500_paired/rlbench_policy_action_diag_rgbd_normal.json
experiments/logs/visible_object_point_scale20_500_strict/rlbench_policy_action_diag_rgbd_normal.json
experiments/logs/visible_object_point_scale20_500_strict/rlbench_policy_action_diag_rgbd_null.json
experiments/logs/visible_object_point_scale20_500_strict/rlbench_policy_action_diag_rgbd_cross_sample.json
```

核心数字：

| diagnostic | normal | null | cross_sample |
|---|---:|---:|---:|
| strict `xyz_rmse` | `0.00502` | `0.00603` | `0.00556` |
| strict `xyz_direction_cosine` | `0.3608` | `0.1140` | `0.2356` |

paired normal-vs-cross：

| metric | value |
|---|---:|
| `paired_depth_point_xyz_l2` | `0.1988` |
| `paired_depth_waypoint_xyz_action_l2` | `0.6776` |
| `paired_pred_xyz_l2` | `0.00743m` |

当时答辩口径：

> OpenVLA 端到端部分还没有 rollout superiority，但最后一次 geometry-bottleneck rescue 已经证明了真实 depth 能影响最终 action，并且 normal depth 在离线 action diagnostic 上优于 null/cross。这个结果不能替代闭环成功，但它推翻了“OpenVLA 里 depth 完全进不了动作”的最坏情况。

最新答辩口径应以后面的 `visible_pre_first_close_point_xyz` 为准：

> OpenVLA 端到端部分还没有 rollout superiority，但最新 visible-precontact gate 已经证明真实 depth 让 selected 3D point 更接近当前可见 pre-contact label，并进入 8-step action chunk。这个结果不能替代闭环成功；它说明瓶颈已经从“depth 是否进入动作”后移到“几何点如何变成可执行 contact trajectory”。

### 6.1b Pre-Contact Target Gate

为对齐 demo-tail upper bound，新增了 `pre_first_close_pose_xyz` auxiliary target。它返回第一次 gripper close 前 `aux_future_horizon` 帧的绝对 EE xyz；当 `aux_future_horizon=20` 时，demo0 正好对应 demo-tail 接管入口。

代码与日志：

```text
vla-scripts/finetune_depthvla.py
prismatic/models/action_heads.py
experiments/robot/rlbench/smoke_rlbench_hdf5_dataset.py
experiments/logs/pre_first_close_smoke_diag/rlbench_policy_action_diag_rgbd_normal.json
```

真实 `open_drawer` label sanity：

| demo | first close index | pre20 index | pre20 xyz | first-close xyz |
|---|---:|---:|---|---|
| demo0 | `74` | `54` | `[0.2118, 0.0601, 1.0512]` | `[0.1961, -0.0205, 1.0356]` |
| demo1 | `79` | `59` | `[0.3162, 0.1424, 1.0666]` | `[0.2978, 0.0273, 1.0357]` |
| demo2 | `77` | `57` | `[0.0953, 0.3066, 1.0284]` | `[0.1296, 0.1497, 1.0357]` |

`MAX_STEPS=1` training smoke：

| check | value |
|---|---:|
| aux target | `pre_first_close_pose_xyz` |
| prediction shape | `(1, 3)` |
| label shape | `(1, 3)` |
| aux spatial loss | `0.0459` |

Small paired diagnostic on the smoke checkpoint:

| metric | value |
|---|---:|
| `paired_depth_point_xyz_l2` | `0.1888` |
| `paired_depth_waypoint_xyz_action_l2` | `0.7335` |
| `paired_depth_waypoint_chunk_xyz_action_l2` | `2.0747` |
| `paired_pred_xyz_l2` | `0.00819m` |

解读：这不是新 rollout 结果，只是把下一轮更合理的 pre-contact geometry target 接进了 OpenVLA 训练和诊断栈。临时 `MAX_STEPS=1` checkpoint 已删除以节省磁盘，保留 JSON 日志。

500-step gate。该 no-go checkpoint 已删除以节省约 `855M` 磁盘，只保留 JSON 诊断日志：

```text
experiments/logs/pre_first_close_scale20_chunk8_500_paired/rlbench_policy_action_diag_rgbd_normal.json
experiments/logs/pre_first_close_scale20_chunk8_500_strict/rlbench_policy_action_diag_rgbd_normal.json
experiments/logs/pre_first_close_scale20_chunk8_500_strict/rlbench_policy_action_diag_rgbd_null.json
experiments/logs/pre_first_close_scale20_chunk8_500_strict/rlbench_policy_action_diag_rgbd_cross_sample.json
```

paired normal-vs-cross：

| metric | value |
|---|---:|
| `paired_depth_point_xyz_l2` | `0.1686` |
| `paired_depth_waypoint_xyz_action_l2` | `0.5694` |
| `paired_depth_waypoint_chunk_xyz_action_l2` | `1.6105` |
| `paired_pred_xyz_l2` | `0.00690m` |

strict normal/null/cross：

| depth mode | `xyz_rmse` | `xyz_direction_cosine` | `pred_xyz_norm` |
|---|---:|---:|---:|
| normal | `0.00382` | `0.588` | `0.00802` |
| null | `0.00491` | `0.251` | `0.00665` |
| cross_sample | `0.00357` | `0.753` | `0.00831` |

解读：

> `pre_first_close_pose_xyz` 是更贴近 demo-tail 入口的 target，并且 normal 明显优于 null；但 cross-sample 在 strict imitation 指标上仍然优于 normal。因此这不是 OpenVLA learned RGB-D success，只能作为“pre-contact target 改善 normal-vs-null，但 object/contact-specific grounding 仍未解决”的证据。

### 6.1c Visible Pre-Contact Point Gate

新增 `visible_pre_first_close_point_xyz`：先取 demo first-close 前 `aux_future_horizon=20` 帧的 EE xyz，再在当前 RGB-D 可见点云中找离它最近的 3D 点作为 label。这比直接监督 EE xyz 更强地绑定当前 depth 几何。

代码与日志：

```text
vla-scripts/finetune_depthvla.py
prismatic/models/action_heads.py
experiments/robot/rlbench/diagnose_policy_actions.py
experiments/robot/rlbench/probe_3d_action_map_feasibility.py
runs_rlbench_visible_preclose_500/
experiments/logs/rlbench_visible_contact_target_visible_preclose_probe.json
experiments/logs/visible_preclose_500_paired/rlbench_policy_action_diag_rgbd_normal.json
experiments/logs/visible_preclose_500_strict/rlbench_policy_action_diag_rgbd_normal.json
experiments/logs/visible_preclose_500_strict/rlbench_policy_action_diag_rgbd_null.json
experiments/logs/visible_preclose_500_strict/rlbench_policy_action_diag_rgbd_cross_sample.json
```

Feasibility probe：

| target | normal median | cross median | EE fallback median |
|---|---:|---:|---:|
| `pre_first_close_pose` | `0.0615m` | `0.0508m` | `0.1084m` |
| `visible_pre_first_close_point` | `0.0000m` | `0.0142m` | `0.0934m` |

500-step gate：

| depth mode | selected point -> aux label L2 | `xyz_rmse` | `xyz_direction_cosine` |
|---|---:|---:|---:|
| normal | `0.099m` | `0.00395` | `0.561` |
| null | `0.699m` | `0.00491` | `0.251` |
| cross_sample | `0.194m` | `0.00364` | `0.722` |

Paired normal-vs-cross：

| metric | value |
|---|---:|
| normal selected point -> aux label | `0.099m` |
| cross selected point -> same aux label | `0.194m` |
| selected-point normal advantage | `0.095m` |
| `paired_depth_point_xyz_l2` | `0.1746m` |
| `paired_depth_waypoint_chunk_xyz_action_l2` | `1.8790` |
| `paired_pred_xyz_l2` | `0.00818m` |

解读：

> 这是目前 OpenVLA/RLBench 端到端部分最清楚的几何正证据：normal depth 让 selected 3D point 更接近当前可见 pre-contact label，明显优于 null 和 cross-sample；同时 selected point 继续进入 waypoint/action chunk。边界仍然是：strict action imitation 指标没有过 cross-sample gate，不能 claim rollout success 或 RGB-D 超过 RGB-only。

Unfrozen action-head follow-up：

```text
runs_rlbench_visible_object_point_scale20_unfrozen_1000/
experiments/logs/visible_object_point_scale20_unfrozen_1000_paired/rlbench_policy_action_diag_rgbd_normal.json
experiments/logs/visible_object_point_scale20_unfrozen_1000_strict/rlbench_policy_action_diag_rgbd_normal.json
experiments/logs/visible_object_point_scale20_unfrozen_1000_strict/rlbench_policy_action_diag_rgbd_null.json
experiments/logs/visible_object_point_scale20_unfrozen_1000_strict/rlbench_policy_action_diag_rgbd_cross_sample.json
```

| diagnostic | normal | null | cross_sample |
|---|---:|---:|---:|
| strict `xyz_rmse` | `0.00377` | `0.00603` | `0.00541` |
| strict `xyz_direction_cosine` | `0.6692` | `0.1140` | `0.2563` |

paired normal-vs-cross：

| metric | value |
|---|---:|
| `paired_depth_point_xyz_l2` | `0.2011` |
| `paired_depth_waypoint_xyz_action_l2` | `0.6293` |
| `paired_pred_xyz_l2` | `0.00690m` |

解读：

> 解冻 action-head base 后，离线 imitation 和 normal/null/cross 分离更强，说明 depth-action coupling 不是假的；但闭环 `open_drawer` 仍失败，并且更早出现 `InvalidActionError`。所以 OpenVLA 最稳妥 claim 是“动作层正向因果诊断”，不是“任务成功率正结果”。

8-step waypoint chunk follow-up：

```text
runs_rlbench_visible_object_point_scale20_chunk8_500/
experiments/logs/visible_object_point_scale20_chunk8_500_paired/rlbench_policy_action_diag_rgbd_normal.json
experiments/logs/visible_object_point_scale20_chunk8_500_strict/rlbench_policy_action_diag_rgbd_normal.json
experiments/logs/visible_object_point_scale20_chunk8_500_strict/rlbench_policy_action_diag_rgbd_null.json
experiments/logs/visible_object_point_scale20_chunk8_500_strict/rlbench_policy_action_diag_rgbd_cross_sample.json
experiments/logs/visible_object_point_scale20_chunk8_500_eval_h200_rpy002/
```

新增代码路径：`DEPTH_WAYPOINT_ACTION_CHUNK_LEN=8` 会把 depth-selected waypoint 写入前 8 个 action chunk 的 xyz，默认仍是 `1`，不改变旧实验。

| diagnostic | normal | null | cross_sample |
|---|---:|---:|---:|
| strict `xyz_rmse` | `0.00414` | `0.00603` | `0.00541` |
| strict `xyz_direction_cosine` | `0.6125` | `0.1140` | `0.2584` |
| strict `depth_waypoint_chunk_xyz_action_norm` | `1.5917` | `1.8765` | `1.8826` |

paired normal-vs-cross：

| metric | value |
|---|---:|
| `paired_depth_point_xyz_l2` | `0.1951` |
| `paired_depth_waypoint_xyz_action_l2` | `0.6024` |
| `paired_depth_waypoint_chunk_xyz_action_l2` | `1.7038` |
| `paired_pred_xyz_l2` | `0.00678m` |

闭环 `MAX_DELTA_RPY=0.02`：

| depth mode | success | length | error | mean xyz step |
|---|---:|---:|---|---:|
| normal | `0/1` | `200` | none | `0.00752` |
| null | `0/1` | `4` | `InvalidActionError` | `0.00484` |
| cross_sample | `0/1` | `190` | `InvalidActionError` | `0.00509` |

真实 chunk execution 评估：

```text
experiments/logs/visible_object_point_scale20_chunk8_500_eval_h200_rpy002_exec8/
experiments/logs/visible_object_point_scale20_chunk8_500_trace_exec8/
```

新增 `ACTION_CHUNK_EXEC_HORIZON=8` 后，eval 会连续执行预测 chunk 的前 8 个动作，再重新预测。结果：

| depth mode | success | length | error | mean xyz step |
|---|---:|---:|---|---:|
| normal | `0/1` | `200` | none | `0.00733` |
| null | `0/1` | `4` | `InvalidActionError` | `0.00480` |

新增 `EVAL_TRACE_OUTPUT` 后，可以导出逐步闭环几何 trace。短 horizon sanity：

| trace metric | normal | null |
|---|---:|---:|
| rows | `32` | `4` |
| chunk index | `0..7` repeated | `0..3` then failure |
| new prediction steps | `0,8,16,24` | `0` |
| rows with depth point | `32/32` | `4/4` |
| mean xyz step | `0.00978` | `0.00480` |

第一步 normal-vs-null trace 差异：

| metric | value |
|---|---:|
| selected depth point L2 | `0.6205m` |
| waypoint chunk L2 | `0.8547` |
| final xyz delta L2 | `0.00655m` |

trace analyzer：

```text
experiments/robot/rlbench/analyze_eval_trace.py
experiments/logs/visible_object_point_scale20_chunk8_500_trace_exec8/trace_analysis.json
experiments/logs/visible_object_point_scale20_chunk8_500_trace_exec8/trace_analysis.md
```

| analysis metric | normal | null |
|---|---:|---:|
| EE displacement | `0.2819m` | `0.0156m` |
| EE-to-depth-point first | `0.2780m` | `0.5589m` |
| EE-to-depth-point last | `0.0144m` | `0.5494m` |
| min EE-to-depth-point | `0.0097m` | `0.5494m` |
| action-depth direction cosine mean | `0.8103` | `0.6714` |
| waypoint saturation fraction | `0.8333` | `0.6667` |
| gripper mean | `0.9141` | `0.9336` |
| first close step | none | none |

eval-only depth-near gripper diagnostic：

```text
experiments/logs/visible_object_point_scale20_chunk8_500_eval_exec8_depthgrip003/
```

使用：

```text
GRIPPER_OVERRIDE_MODE=latch_close_near_depth_point
GRIPPER_CLOSE_DISTANCE=0.03
ACTION_CHUNK_EXEC_HORIZON=8
```

| depth mode | success | length | error | first close step | gripper mean |
|---|---:|---:|---|---:|---:|
| normal | `0/1` | `200` | none | `27` | `0.135` |
| null | `0/1` | `4` | `InvalidActionError` | none | `1.000` |

解读：normal 的 gripper 确实在接近 selected depth point 后 latch close，最小 EE-to-depth-point 距离 `0.00464m`，但仍未完成 open drawer。

解读：

> chunk8 版本把 OpenVLA depth 几何通道从 first-step xyz 扩展到 8-step action chunk，并且真实执行 chunk 时 normal 仍跑满、null 仍快速失败。trace analyzer 进一步说明 normal 不是随机稳定：EE 到 selected depth point 的距离从 `0.278m` 降到 `0.014m`，动作方向和 depth point 方向 cosine 均值 `0.8103`。depth-near gripper 诊断又说明，即使用 depth 几何触发夹爪关闭，任务仍未成功；主要剩余问题更像 contact point/拉开轨迹/后接触动作，而不是 depth 没进入 temporal action geometry。

### 6.4 最后 OpenVLA contact-target 诊断

最后补了一个更强的 contact-style target 和 post-contact eval diagnostic：

- 代码入口：`first_close_pose_xyz` auxiliary target。
- 训练 run：

```text
runs_rlbench_first_close_scale20_chunk8_500/
```

- paired diagnostic：

```text
experiments/logs/first_close_scale20_chunk8_500_paired/rlbench_policy_action_diag_rgbd_normal.json
```

| metric | value |
|---|---:|
| `paired_depth_point_xyz_l2` | `0.1594` |
| `paired_depth_waypoint_xyz_action_l2` | `0.5739` |
| `paired_depth_waypoint_chunk_xyz_action_l2` | `1.6233` |
| `paired_pred_xyz_l2` | `0.00659m` |
| normal selected point mean z | `1.102` |

Post-close pull 诊断：

```text
experiments/logs/visible_object_point_scale20_chunk8_500_eval_exec8_depthgrip003_pullY001/
experiments/logs/first_close_scale20_chunk8_500_eval_exec8_depthgrip003_pullY001/
experiments/logs/first_close_scale20_chunk8_500_eval_exec8_depthgrip003_pullY001_rpy0/
experiments/logs/first_close_scale20_chunk8_500_eval_exec8_depthgrip020_pullY001_rpy0/
experiments/logs/first_close_scale20_chunk8_500_eval_latched_first_pullY001/
experiments/logs/first_close_scale20_chunk8_500_eval_oracle_first_close_pullY001/
experiments/logs/first_close_scale20_chunk8_500_eval_oracle_first_close_grip005_pullY001/
experiments/logs/demo_replay_open_drawer_3episodes/
experiments/logs/first_close_scale20_chunk8_500_eval_oracle_first_close_demotail_pre5/
experiments/logs/first_close_scale20_chunk8_500_eval_oracle_first_close_demotail_pre5_grip005/
experiments/logs/first_close_scale20_chunk8_500_eval_oracle_first_close_demotail_pre20_grip008/
experiments/logs/first_close_scale20_chunk8_500_eval_oracle_first_close_demotail_pre20_grip020/
experiments/logs/first_close_scale20_chunk8_500_eval_learned_first_demotail_pre20_grip020_normal/
experiments/logs/first_close_scale20_chunk8_500_eval_learned_first_demotail_pre20_grip020_null/
experiments/logs/first_close_scale20_chunk8_500_eval_learned_first_demotail_pre20_grip020_cross/
experiments/logs/first_close_scale20_chunk8_500_eval_learned_first_demotail_pre20_grip020_shuffle/
experiments/logs/first_close_scale20_chunk8_500_eval_learned_first_demotail_pre20_grip0195_normal/
experiments/logs/first_close_scale20_chunk8_500_eval_learned_first_demotail_pre20_grip0195_cross/
```

| setting | success | length | error | trace conclusion |
|---|---:|---:|---|---|
| visible-object target + close at `0.03m` + pull `+Y` | `0/1` | `38` | `InvalidActionError` | gripper closes and pulls `10` steps, but close point is too high (`z≈1.26`) |
| first-close target + `MAX_DELTA_RPY=0.02` | `0/1` | `15` | `InvalidActionError` | contact target makes early approach more aggressive |
| first-close target + `MAX_DELTA_RPY=0` + close `0.03m` | `0/1` | `61` | `InvalidActionError` | no close; min EE-to-depth-point is `0.1815m` |
| first-close target + `MAX_DELTA_RPY=0` + close `0.2m` | `0/1` | `61` | `InvalidActionError` | closes at step `59`, pulls `1` step, still fails |
| latch first selected point + close `0.03m` + pull `+Y` | `0/1` | `200` | none | fixed point reaches min `0.00062m`, closes at step `45`, pulls `35` steps |
| oracle demo first-close point + close `0.03m` | `0/1` | `200` | none | reaches min `0.039m`, no close |
| oracle demo first-close point + close `0.05m` | `0/1` | `200` | none | still no close/open in this rollout |
| direct demo replay, same action mode | `3/3` | mean `92.33` | none | stored expert EE pose/gripper sequence solves `open_drawer` in the current eval stack |
| oracle first-close point + demo tail, close `0.03m`, preclose `5` | `0/1` | `220` | none | demo tail not triggered; min distance `0.0409m` |
| oracle first-close point + demo tail, close `0.05m`, preclose `5` | `0/1` | `220` | none | demo tail not triggered; min distance `0.0578m` |
| oracle first-close point + demo tail, close `0.08m`, preclose `20` | `0/1` | `220` | none | demo tail not triggered; min distance `0.1038m` |
| oracle first-close point + demo tail, close `0.20m`, preclose `20` | `1/1` | `61` | none | tail activates at step `25`, demo index `54`, distance to first-close point `0.1916m` |
| learned first point + demo tail, normal, close `0.20m` | `1/1` | `64` | none | learned first selected point `[0.396,0.140,1.039]`; tail step `28`; min distance `0.192m` |
| learned first point + demo tail, null, close `0.20m` | `0/1` | `2` | `InvalidActionError` | null selected point `[0.827,-0.003,1.577]`; no tail |
| learned first point + demo tail, cross, close `0.20m` | `1/1` | `60` | none | cross point near normal, `[0.398,0.119,1.071]`; tail step `24` |
| learned first point + demo tail, shuffle, close `0.20m` | `1/1` | `61` | none | shuffled point also near handle-height region, `[0.386,0.149,1.069]`; tail step `25` |
| learned first point + demo tail, normal/cross, close `0.195m` | `1/1` / `1/1` | `64` / `61` | none | tighter threshold still does not separate normal from cross |

Trace 解释：

- `first_close_pose_xyz` 的第一步 selected point 已接近把手高度，`z≈1.039`。
- 但闭环中 selected point drift 为 `0.3408m`，后续点漂到 `z≈0.83-0.88` 的错误区域。
- EE-to-selected-point 方向 cosine 均值约 `0.94`，说明 policy 仍在跟随 depth geometry；失败边界是 selected contact point 不稳定、非 object-bound，而不是 depth 没进入 action。
- latch-first 诊断把 selected point drift 降为 `0`，EE-to-point 从 `0.473m` 降到 `0.001m` 量级并跑满 `200` 步，但仍未成功；因此 point drift 是稳定性问题，但不是唯一问题。
- oracle demo first-close point 诊断也未成功，说明还需要正确 gripper orientation/contact constraints 和 post-contact trajectory，而不是只要给一个 xyz 接触点。
- direct demo replay `3/3` 排除了“当前 RLBench action mode 无法执行专家轨迹”的解释。
- demo-tail upper bound 在 `0.20m` 宽阈值下成功，说明一旦进入足够接近的接触前邻域，专家 temporal/contact trajectory 可以接上；但 `0.03/0.05/0.08m` 均未触发，说明当前直线 point controller 还不能稳定进入严格接触状态。
- learned first selected point + demo tail 的 normal/null 对比说明，完全移除几何会选到离谱空间点并快速 `InvalidActionError`；但 cross/shuffle 也能触发并完成，说明这个 wide-gate oracle 不能作为 strict normal-vs-corrupt success。

闭环 sanity rollout 仍未通过：

```text
experiments/logs/visible_object_point_scale20_500_eval_h200/rgbd_normal.json
experiments/logs/visible_object_point_scale20_500_eval_h200_rpy002/
experiments/logs/visible_object_point_scale20_500_eval_h200_wpscale30_rpy002/rgbd_normal.json
experiments/logs/visible_object_point_scale20_500_eval_h200_rpy002_grip75/rgbd_normal.json
```

无 rpy clamp 时 normal 是 `0/1`，length `193`，`InvalidActionError`，平均 xyz 步长 `0.00865`。

加入 eval-only `MAX_DELTA_RPY=0.02` 后：

| setting | success | length | error | mean xyz step |
|---|---:|---:|---|---:|
| normal | `0/1` | `200` | none | `0.00811` |
| null | `0/1` | `4` | `InvalidActionError` | `0.00486` |
| cross_sample | `0/1` | `200` | none | `0.00489` |
| normal, eval `wpscale=30` | `0/1` | `200` | none | `0.00952` |
| normal, gripper close after step `75` | `0/1` | `200` | none | `0.00839` |

Unfrozen action-head 闭环：

| setting | success | length | error | mean xyz step |
|---|---:|---:|---|---:|
| normal, `MAX_DELTA_RPY=0.02` | `0/1` | `67` | `InvalidActionError` | `0.00877` |
| normal, `MAX_DELTA_RPY=0.005` | `0/1` | `82` | `InvalidActionError` | `0.00984` |
| normal, `MAX_DELTA_RPY=0.005`, gripper forced open | `0/1` | `82` | `InvalidActionError` | similar |

Gripper 诊断：

- demo gripper 在约第 `73-78` 步从 open=`1` 切到 close=`0`。
- learned normal rollout 平均 gripper command `0.888`，偏 open。
- eval-only close-after-75 后平均 gripper command 降到 `0.375`，但仍 `0/1`。
- 因此 gripper timing 是问题之一，但不是唯一解释；闭环 success 还需要 temporal/action decoder 能形成真正的抓取-拉开轨迹。

因此 OpenVLA 部分最终应表述为：

> 端到端 action diagnostic 正向；闭环 normal/null 有稳定性分离；但 rollout task success 和 RGB-D superiority 仍未完成。

这支撑最终解释：

> depth 不是简单加到任何 VLA 上都会有用；它需要 3D-sensitive benchmark、object-centric representation 和 primary/temporal action decoder。

## 7. 一键汇总

```bash
python scripts/collect_depthvla_final_results.py
```

输出：

```text
FINAL_RESULTS_TABLE.md
experiments/logs/final_results_table.csv
```

## 8. 答辩一句话

> 我们最终证明了 depth/pointcloud 在 ManiSkill3 PickCube 上可以通过 learned policy 转成闭环收益：normal pointcloud `20/60`，eval-time null/cross `1/60`，matched sampled-RGB-only baseline `1/60`，matched null/proprio baseline `3/60`。所以结论不是 depth 没用，而是 depth 必须进入合适的 3D-sensitive benchmark 和 primary pointcloud/action decoder。

## 9. 官方来源核对

联网复核和本地结果的对应关系见：

```text
FINAL_SOURCE_AUDIT.md
```

核心口径：

> ManiSkill3 / RLBench / CALVIN 解释了为什么要换 benchmark 和比较 absolute-vs-relative action；DP3 / PerAct / Act3D / BridgeVLA / PointVLA / SpatialVLA 共同指向同一个方法结论：depth/pointcloud 必须和 action space 对齐。我们的 ManiSkill3 `20/60` vs `1/60` / `3/60` 正结果正是在这个方向上成立的。
