# OpenVLA 端到端 Geometry Bottleneck 继续路线

更新时间：2026-07-05 UTC

## 1. 当前状态

这个文件只记录 OpenVLA 端到端路线，不把 ManiSkill3 teacher-distilled pointcloud 结果当成 OpenVLA 成功。

最新结论更新：

> `visible_pre_first_close_point_xyz + 8-step waypoint action chunk` 是目前 OpenVLA/RLBench 端到端部分最清楚的几何正证据：真实 normal depth 让 selected 3D point 更接近当前可见 pre-contact label，优于 null 和 cross-sample，并且这个 selected point 被写入 8-step waypoint/action chunk。

这仍然不是 rollout 成功，也还不能 claim RGB-D 超过 RGB-only；因为 strict action imitation 里 cross-sample 的 `xyz_rmse` 和 cosine 仍优于 normal。它能说明的是：depth 已经进入 OpenVLA 的几何 bottleneck 和 temporal action chunk，但几何点到可执行 contact trajectory 的 action decoder 仍没有解决。

最后补跑的 `pre_first_close_pose_xyz` 和 `visible_pre_first_close_point_xyz` gate 把 target 对齐到 demo-tail 成功接管的 pre-contact 入口。前者让 normal 优于 null，但 cross-sample 仍能追上；后者进一步把 label 绑定到当前 RGB-D 可见点云，selected-point geometry 通过 normal/null/cross gate。因此 OpenVLA/RLBench 的最终边界是：depth-action coupling 已经不是完全缺失，剩余问题是 object/contact-specific grounding、gripper pose/orientation 和 post-contact temporal trajectory。

当前已经完成的代码级推进：

- 新增 `visible_object_point_xyz` auxiliary target：从当前 RGB-D 的 agentview depth 反投影出可见几何点，并监督 dense point selector 选择绝对 3D 点。
- 新增 `visible_object_rel_xyz` auxiliary target：监督 object-query head 预测 EE-relative 可见物体几何向量。
- 新增 `pre_first_close_pose_xyz` auxiliary target：监督第一次 gripper close 前 `aux_future_horizon` 帧的绝对 EE xyz；`aux_future_horizon=20` 时，它直接对齐 demo-tail 成功接管的 pre-contact 入口帧。
- 新增 `visible_first_close_point_xyz` 和 `visible_pre_first_close_point_xyz` auxiliary target：把 demo contact/pre-contact xyz 投影到当前 RGB-D 可见点云中最近的真实 3D 点，减少直接拟合 demo EE 坐标的 shortcut。
- `visible_object_point_xyz`、`visible_first_close_point_xyz` 和 `visible_pre_first_close_point_xyz` 被接入 `L1RegressionActionHead.predict_spatial_delta()` 的 point-selection 路径。
- 诊断脚本会记录 selected 3D point / waypoint action，并支持 normal-vs-cross 的 paired geometry delta。
- 诊断脚本会记录 selected point 到 auxiliary label 的误差，用来判断 normal depth 是否真的选到了当前可见 pre-contact 几何。
- 新增 smoke：`experiments/robot/rlbench/smoke_depth_geometry_bottleneck.py`，验证 selected point 能通过 waypoint override 写入 first-step xyz action。
- 新增 `depth_waypoint_action_scale`：先把米制 waypoint delta clip 到 `0.02m`，再映射到 OpenVLA normalized action 的合适量级，避免 `0.02m` 被误当作 `0.02` 个 normalized unit 后反归一化成 `0.00035m`。
- 新增 eval-only `max_delta_rpy`：闭环 rollout 时可以裁剪每步 rpy 增量，避免姿态累积把 RLBench planning 推到 `InvalidActionError`。默认不启用，不影响离线 diagnostic。

这一步的意义：

> OpenVLA action path 现在有了一个更接近 ManiSkill 成功经验的几何瓶颈：不是直接预测 future pose，而是先从 depth point tokens 中选出可见 object/contact-like geometry，再把这个 EE-relative 几何作为 primary xyz action。`pre_first_close_pose_xyz` 进一步把目标从“闭合点”前移到“可接 demo-tail 的 pre-contact 入口”，更符合最后 oracle upper bound 暴露出的需求。

## 2. 过程证据：Visible Object Gate

配置：

```text
TASKS=open_drawer
DATASET_NAME=rlbench_open_drawer_visible_object_point
HDF5_DIR=/root/RLBench/rgbd_hdf5_open_drawer_3demos_64
DEPTH_AUX_TARGET=visible_object_point_xyz
DEPTH_AUX_OUTPUT_DIM=3
DEPTH_AUX_SPATIAL_LOSS_WEIGHT=0.2
DEPTH_WAYPOINT_ACTION_WEIGHT=1.0
DEPTH_WAYPOINT_ACTION_CLIP=0.02
DEPTH_WAYPOINT_ACTION_SCALE=20.0
MAX_STEPS=500
FREEZE_VLA_LORA=True
FREEZE_PROPRIO_PROJECTOR=True
FREEZE_ACTION_HEAD_BASE=True
```

checkpoint：

```text
/root/autodl-tmp/openvla-oft/runs_rlbench_visible_object_point_scale20_500/47a0ec7fc4ec123775a391911046cf33cf9ed83f+rlbench_open_drawer_visible_object_point+depth-densep1024+object-query+gate-1.0+wpact-1.0+wpclip-0.02+wpscale-20.0+aux-visible_object_point_xyz-0.2+b1+lr-0.0001+lora-r4+dropout-0.0--rlbench-rgbd-dense-keypose
```

paired normal-vs-cross diagnostic：

| metric | value |
|---|---:|
| `paired_depth_point_xyz_l2` | `0.1988` |
| `paired_depth_waypoint_xyz_action_l2` | `0.6776` |
| `paired_pred_xyz_l2` | `0.00743m` |
| `pred_xyz_norm` | `0.00729m` |
| `target_xyz_norm` | `0.00853m` |

strict normal/null/cross-sample action diagnostic：

| depth mode | `xyz_rmse` | `xyz_direction_cosine` | `pred_xyz_norm` |
|---|---:|---:|---:|
| normal | `0.00502` | `0.3608` | `0.00729` |
| null | `0.00603` | `0.1140` | `0.00669` |
| cross_sample | `0.00556` | `0.2356` | `0.00822` |

解读：

> 真实 depth 不只改变了中间 selected point，也通过 waypoint override 改变了最终反归一化 action；并且 normal depth 在 demo-action imitation 上优于 null 和 cross-sample。这是 OpenVLA 端到端 RGB-D 路线目前最强的正向诊断结果。

闭环 sanity rollout：

```text
experiments/logs/visible_object_point_scale20_500_eval_h200/rgbd_normal.json
experiments/logs/visible_object_point_scale20_500_eval_h200_rpy002/
experiments/logs/visible_object_point_scale20_500_eval_h200_wpscale30_rpy002/rgbd_normal.json
experiments/logs/visible_object_point_scale20_500_eval_h200_rpy002_grip75/rgbd_normal.json
```

结果：

| setting | success | length | error | mean xyz step | mean rpy step |
|---|---:|---:|---|---:|---:|
| normal, no rpy clamp | `0/1` | `193` | `InvalidActionError` | `0.00865` | `0.0337` |
| normal, `MAX_DELTA_RPY=0.02` | `0/1` | `200` | none | `0.00811` | `0.0281` |
| null, `MAX_DELTA_RPY=0.02` | `0/1` | `4` | `InvalidActionError` | `0.00486` | `0.0308` |
| cross, `MAX_DELTA_RPY=0.02` | `0/1` | `200` | none | `0.00489` | `0.0279` |
| normal, eval `wpscale=30`, `MAX_DELTA_RPY=0.02` | `0/1` | `200` | none | `0.00952` | `0.0270` |
| normal, `MAX_DELTA_RPY=0.02`, gripper close after step `75` | `0/1` | `200` | none | `0.00839` | `0.0269` |

因此最终边界是：

> 离线 causal action gate 已通过；加 rpy clamp 后 normal depth 可以稳定跑满 200 步，null depth 会在第 4 步失败，说明真实 depth 已经影响闭环控制稳定性。但 closed-loop success 仍然是 `0/1`，剩余问题更像 gripper/temporal action decoder 和任务完成策略，而不是 depth 完全无法进入 OpenVLA action。

额外 gripper 诊断：

> open_drawer demo 在约第 `73-78` 步从 gripper open=`1` 切到 close=`0`。模型 normal rollout 的平均 gripper command 是 `0.888`，明显偏 open；但 eval-only gripper close-after-75 把平均 gripper 降到 `0.375` 后仍然 `0/1`。所以 gripper 是问题之一，但不是唯一瓶颈；还需要更好的 temporal/action decoder 或 object/contact-conditioned waypoint sequence。

### Unfrozen action-head follow-up

为了让 rpy/gripper/chunk action 不再被 frozen base 限制，又跑了一轮只解冻 action-head base 的训练：

```text
RUN_ROOT_DIR=/root/autodl-tmp/openvla-oft/runs_rlbench_visible_object_point_scale20_unfrozen_1000
MAX_STEPS=1000
FREEZE_VLA_LORA=True
FREEZE_PROPRIO_PROJECTOR=True
FREEZE_ACTION_HEAD_BASE=False
```

checkpoint：

```text
/root/autodl-tmp/openvla-oft/runs_rlbench_visible_object_point_scale20_unfrozen_1000/47a0ec7fc4ec123775a391911046cf33cf9ed83f+rlbench_open_drawer_visible_object_point+depth-densep1024+object-query+gate-1.0+wpact-1.0+wpclip-0.02+wpscale-20.0+aux-visible_object_point_xyz-0.2+b1+lr-0.0001+lora-r4+dropout-0.0--rlbench-rgbd-dense-keypose
```

paired normal-vs-cross diagnostic：

| metric | value |
|---|---:|
| `paired_depth_point_xyz_l2` | `0.2011` |
| `paired_depth_waypoint_xyz_action_l2` | `0.6293` |
| `paired_pred_xyz_l2` | `0.00690m` |
| `xyz_rmse` | `0.00377m` |
| `xyz_direction_cosine` | `0.6692` |

strict normal/null/cross-sample action diagnostic：

| depth mode | `xyz_rmse` | `xyz_direction_cosine` | `pred_xyz_norm` |
|---|---:|---:|---:|
| normal | `0.00377` | `0.6692` | `0.00694` |
| null | `0.00603` | `0.1140` | `0.00669` |
| cross_sample | `0.00541` | `0.2563` | `0.00827` |

闭环 rollout：

| setting | success | length | error | mean xyz step | mean rpy step | mean gripper |
|---|---:|---:|---|---:|---:|---:|
| normal, `MAX_DELTA_RPY=0.02` | `0/1` | `67` | `InvalidActionError` | `0.00877` | `0.0306` | `0.760` |
| normal, `MAX_DELTA_RPY=0.005` | `0/1` | `82` | `InvalidActionError` | `0.00984` | `0.00866` | `0.754` |
| normal, `MAX_DELTA_RPY=0.005`, gripper forced open | `0/1` | `82` | `InvalidActionError` | similar | similar | `1.000` |

解读：

> Unfrozen action-head 明显改善了离线动作拟合和 normal/null/cross 分离，但没有改善闭环成功，反而更早触发 `InvalidActionError`。gripper forced-open 仍然在第 `82` 步失败，说明这不是单纯 gripper close 的问题。最终 OpenVLA 结论应保持为：depth 已经能进入 final action，并在离线 action diagnostic 上优于 corrupt controls；但闭环任务成功还需要更强 temporal/action decoder 或显式 contact/trajectory supervision。

### 8-step waypoint chunk follow-up

因为前面的 waypoint 只覆盖 `action[:, 0, :3]`，又补了一版 temporal geometry path：新增 `DEPTH_WAYPOINT_ACTION_CHUNK_LEN`，默认仍为 `1`，设为 `8` 时会把同一个 depth-selected waypoint 写入前 8 个 action chunk 的 xyz，不碰 rpy/gripper。

配置：

```text
RUN_ROOT_DIR=/root/autodl-tmp/openvla-oft/runs_rlbench_visible_object_point_scale20_chunk8_500
RUN_ID_NOTE=c8
MAX_STEPS=500
RESUME_COMPONENTS_FROM=<scale20_500 checkpoint>
FREEZE_VLA_LORA=True
FREEZE_PROPRIO_PROJECTOR=True
FREEZE_ACTION_HEAD_BASE=True
DEPTH_WAYPOINT_ACTION_CHUNK_LEN=8
```

checkpoint：

```text
/root/autodl-tmp/openvla-oft/runs_rlbench_visible_object_point_scale20_chunk8_500/47a0ec7fc4ec123775a391911046cf33cf9ed83f+rlbench_open_drawer_visible_object_point+depth-densep1024+object-query+gate-1.0+wpact-1.0+wpclip-0.02+wpscale-20.0+wpchunk-8+aux-visible_object_point_xyz-0.2+b1+lr-0.0001+lora-r4+dropout-0.0--c8
```

paired normal-vs-cross diagnostic：

| metric | value |
|---|---:|
| `paired_depth_point_xyz_l2` | `0.1951` |
| `paired_depth_waypoint_xyz_action_l2` | `0.6024` |
| `paired_depth_waypoint_chunk_xyz_action_l2` | `1.7038` |
| `paired_pred_xyz_l2` | `0.00678m` |

strict normal/null/cross-sample action diagnostic：

| depth mode | `xyz_rmse` | `xyz_direction_cosine` | `depth_waypoint_chunk_xyz_action_norm` |
|---|---:|---:|---:|
| normal | `0.00414` | `0.6125` | `1.5917` |
| null | `0.00603` | `0.1140` | `1.8765` |
| cross_sample | `0.00541` | `0.2584` | `1.8826` |

闭环 rollout，`MAX_DELTA_RPY=0.02`：

| depth mode | success | length | error | mean xyz step | mean gripper |
|---|---:|---:|---|---:|---:|
| normal | `0/1` | `200` | none | `0.00752` | `0.899` |
| null | `0/1` | `4` | `InvalidActionError` | `0.00484` | `1.000` |
| cross_sample | `0/1` | `190` | `InvalidActionError` | `0.00509` | `0.909` |

真实 chunk execution sanity：

之前 eval 默认每步重新预测，只执行 `action_chunk[0]`。现在新增 `ACTION_CHUNK_EXEC_HORIZON`，默认 `1`；设为 `8` 时会预测一次 chunk，连续执行前 8 个动作，再重新观测和预测。

| depth mode | `ACTION_CHUNK_EXEC_HORIZON` | success | length | error | mean xyz step |
|---|---:|---:|---:|---|---:|
| normal | `8` | `0/1` | `200` | none | `0.00733` |
| null | `8` | `0/1` | `4` | `InvalidActionError` | `0.00480` |

闭环 trace sanity：

新增 `EVAL_TRACE_OUTPUT`，可导出每步 JSONL，包含 EE pose、delta action、target xyz、gripper、depth-selected point、当前 chunk waypoint。短 horizon trace：

```text
experiments/logs/visible_object_point_scale20_chunk8_500_trace_exec8/rgbd_normal_trace.jsonl
experiments/logs/visible_object_point_scale20_chunk8_500_trace_exec8/rgbd_null_trace.jsonl
```

trace 检查：

| metric | normal | null |
|---|---:|---:|
| trace rows | `32` | `4` |
| chunk index pattern | `0..7` repeated | `0..3` then failure |
| new prediction steps | `0,8,16,24` | `0` |
| rows with depth point | `32/32` | `4/4` |
| mean xyz step | `0.00978` | `0.00480` |

第一步 normal-vs-null 差异：

| metric | value |
|---|---:|
| selected depth point L2 | `0.6205m` |
| waypoint chunk L2 | `0.8547` |
| final xyz delta L2 | `0.00655m` |

新增 trace analyzer：

```text
experiments/robot/rlbench/analyze_eval_trace.py
experiments/logs/visible_object_point_scale20_chunk8_500_trace_exec8/trace_analysis.json
experiments/logs/visible_object_point_scale20_chunk8_500_trace_exec8/trace_analysis.md
```

核心分析：

| metric | normal | null |
|---|---:|---:|
| EE displacement in trace | `0.2819m` | `0.0156m` |
| EE-to-depth-point first | `0.2780m` | `0.5589m` |
| EE-to-depth-point last | `0.0144m` | `0.5494m` |
| min EE-to-depth-point | `0.0097m` | `0.5494m` |
| action-depth direction cosine mean | `0.8103` | `0.6714` |
| waypoint saturation fraction | `0.8333` | `0.6667` |
| gripper mean | `0.9141` | `0.9336` |
| first close step | none | none |

depth-near gripper diagnostic：

新增 eval-only gripper 模式：

```text
GRIPPER_OVERRIDE_MODE=latch_close_near_depth_point
GRIPPER_CLOSE_DISTANCE=0.03
```

它在 EE 距 selected depth point 小于阈值后关闭并保持关闭。结果：

```text
experiments/logs/visible_object_point_scale20_chunk8_500_eval_exec8_depthgrip003/
```

| depth mode | success | length | error | first close step | gripper mean |
|---|---:|---:|---|---:|---:|
| normal | `0/1` | `200` | none | `27` | `0.135` |
| null | `0/1` | `4` | `InvalidActionError` | none | `1.000` |

normal trace 中 gripper 确实在 step `27` latch close，最小 EE-to-depth-point 距离为 `0.00464m`，但任务仍未成功。

解读：

> 这版没有解决 task success，但把结论推进了一步：depth-selected geometry 不只影响 first-step xyz，而是能被写入并真实执行 8-step action chunk；normal trace 中 EE 明显朝 selected depth point 靠近，距离从 `0.278m` 降到 `0.014m`，动作方向和 depth point 方向高度一致。进一步用 depth-near gate 修正 gripper timing 后仍然不成功，所以剩余短板已经更明确：不是“没关夹爪”，而是 contact point/拉开轨迹/后接触动作仍不对。

### 2.6 最后 contact-target 尝试：first-close pose

为确认 visible-object point 是否选错了可操作接触点，最后补了两个 eval-only/训练诊断入口：

- `POST_CLOSE_PULL_DELTA_XYZ` / `POST_CLOSE_PULL_STEPS`：夹爪 latch close 后，沿 demo 主拉开方向执行固定小位移。默认关闭，只用于诊断。
- `first_close_pose_xyz` auxiliary target：用 demo 中第一次 gripper close 的 EE xyz 作为接触点监督，避免 `visible_object_point_xyz` 选到非把手的普通可见点。

关键日志：

```text
experiments/logs/visible_object_point_scale20_chunk8_500_eval_exec8_depthgrip003_pullY001/
experiments/logs/first_close_scale20_chunk8_500_paired/
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

结果：

| setting | success | length | error | key observation |
|---|---:|---:|---|---|
| visible-object + depth-near close + pull `+Y` | `0/1` | `38` | `InvalidActionError` | close 后 pull 了 `10` 步，但闭合高度约 `z=1.26`，高于 demo 接触高度 `z≈1.035` |
| first-close target, paired normal-vs-cross | n/a | n/a | n/a | `paired_pred_xyz_l2=0.00659m`，`paired_depth_point_xyz_l2=0.159`，平均 selected point `z=1.102` |
| first-close target, `MAX_DELTA_RPY=0.02` | `0/1` | `15` | `InvalidActionError` | contact target 太激进，早期规划失败 |
| first-close target, `MAX_DELTA_RPY=0` | `0/1` | `61` | `InvalidActionError` | 更稳定，但没有进入 `0.03m` close gate |
| first-close target, close distance `0.2`, `MAX_DELTA_RPY=0` | `0/1` | `61` | `InvalidActionError` | 第 `59` 步闭合，第 `60` 步 pull 1 步后失败 |
| latch learned first selected point + pull | `0/1` | `200` | none | selected point drift 变为 `0`，step `45` 关闭，pull `35` 步，仍未成功 |
| oracle demo first-close point + close `0.03` | `0/1` | `200` | none | 朝 demo close point 走，min distance `0.039m`，未触发关闭 |
| oracle demo first-close point + close `0.05` | `0/1` | `200` | none | 仍未打开，说明 oracle contact point 加直线接近/固定拉动也不充分 |
| direct demo replay, same action mode | `3/3` | mean `92.33` | none | 当前 RLBench eval stack 能执行存储专家 EE pose/gripper 轨迹 |
| oracle first-close point + demo tail, close `0.03`, preclose `5` | `0/1` | `220` | none | demo tail 未触发，min distance `0.0409m` |
| oracle first-close point + demo tail, close `0.05`, preclose `5` | `0/1` | `220` | none | demo tail 未触发，min distance `0.0578m` |
| oracle first-close point + demo tail, close `0.08`, preclose `20` | `0/1` | `220` | none | demo tail 未触发，min distance `0.1038m` |
| oracle first-close point + demo tail, close `0.20`, preclose `20` | `1/1` | `61` | none | step `25` 接管专家 tail，demo index `54`，接管时距 first-close point `0.1916m` |
| learned first point + demo tail, normal, close `0.20` | `1/1` | `64` | none | 模型第一步 selected point 为 `[0.396,0.140,1.039]`，tail step `28` |
| learned first point + demo tail, null, close `0.20` | `0/1` | `2` | `InvalidActionError` | null selected point 为 `[0.827,-0.003,1.577]`，未触发 tail |
| learned first point + demo tail, cross, close `0.20` | `1/1` | `60` | none | cross selected point 和 normal 接近，tail step `24` |
| learned first point + demo tail, shuffle, close `0.20` | `1/1` | `61` | none | shuffle selected point 仍落在 handle-height 附近，tail step `25` |
| learned first point + demo tail, normal/cross, close `0.195` | `1/1` / `1/1` | `64` / `61` | none | 收紧阈值仍不能分离 normal 和 cross |

最重要的 trace 现象：

- first-close target 第一步 selected depth point 已经到把手高度附近：`z≈1.039`。
- 但闭环中 selected point drift 达到 `0.3408m`，后续点逐渐漂到 `z≈0.83-0.88` 的错误接触区域。
- EE 始终朝 selected point 走，方向 cosine 均值 `0.94` 左右；失败不是 depth 没进动作，而是 selected point 不够 object-bound / contact-stable。
- latch learned first selected point 后，EE 能把距离从 `0.473m` 降到 `0.00062m`，且不会再 `InvalidActionError`；但这个点的 x/y 不是 demo0 的真实 contact，因此抓取/拉动仍无效。
- oracle demo first-close point 也没有打开抽屉，说明剩余问题还包括 EE pose/orientation、实际接触约束和 post-contact trajectory，不只是点选择漂移。
- direct demo replay `3/3` 说明同一个 `MoveArmThenGripper(ClippedEndEffectorPoseViaPlanning(), Discrete())` action mode 可以执行专家轨迹；失败不能归因于 eval stack 根本不可用。
- demo-tail upper bound 在 `0.20m` 宽阈值下成功，说明“进入接触前邻域 + 正确 temporal/contact tail”足以完成任务；但 `0.03/0.05/0.08m` 都没有触发，说明当前直线 xyz point controller 还不能稳定进入严格接触状态。
- learned first selected point + demo tail 的 normal/null 对比有用：null 会选到离谱点并在第 `2` 步 planner failure；但 cross/shuffle 也能接上 demo tail，所以这条不是 strict normal-vs-corrupt success，只能说明当前 selected-point bottleneck 还不够 object/contact-specific。

最终边界：

> `first_close_pose_xyz` 证明 contact-style supervision 比 generic visible point 更接近任务需求；latch 诊断进一步证明固定 selected point 可以消除漂移并稳定闭环。demo replay 和 demo-tail upper bound 又说明任务/控制栈本身可解，但 learned OpenVLA policy 还没有稳定进入严格、object-bound 的接触前状态，也没有学到后续 temporal/contact trajectory。因此下一步不能只改 point selector，还需要显式 handle/contact pose、gripper orientation 和 post-contact action decoder。

### Pre-Contact Target Gate

为了把 target 对齐到 demo-tail upper bound 的入口，新加入 `pre_first_close_pose_xyz`：

```text
DEPTH_AUX_TARGET=pre_first_close_pose_xyz
DEPTH_AUX_OUTPUT_DIM=3
DEPTH_AUX_FUTURE_HORIZON=20
```

真实 `open_drawer` sanity：

| demo | first close index | pre20 index | pre20 xyz |
|---|---:|---:|---|
| demo0 | `74` | `54` | `[0.2118, 0.0601, 1.0512]` |
| demo1 | `79` | `59` | `[0.3162, 0.1424, 1.0666]` |
| demo2 | `77` | `57` | `[0.0953, 0.3066, 1.0284]` |

验证状态：

- `smoke_rlbench_hdf5_dataset.py` 通过，`pre_first_close_pose_xyz_aux` shape 为 `(3,)`。
- `MAX_STEPS=1` 真实训练 smoke 通过，aux prediction/label shape 都是 `(1,3)`，aux spatial loss finite。
- 小样本 paired diagnostic 能加载该 smoke checkpoint 并输出 `paired_depth_point_xyz_l2=0.1888`、`paired_pred_xyz_l2=0.00819m`。
- 临时 smoke checkpoint 已删除，只保留 `experiments/logs/pre_first_close_smoke_diag/rlbench_policy_action_diag_rgbd_normal.json`。

500-step 训练与诊断已经完成。该 no-go checkpoint 已删除以节省约 `855M` 磁盘，只保留 JSON 诊断日志：

```text
experiments/logs/pre_first_close_scale20_chunk8_500_paired/
experiments/logs/pre_first_close_scale20_chunk8_500_strict/
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

结论：

> `pre_first_close_pose_xyz` 是更合理的 pre-contact target，并且 normal 明显优于 null；但 cross-sample 仍然在 strict imitation 指标上优于 normal。它没有给出 OpenVLA learned task success，也没有证明 RGB-D 超过 RGB-only。后续不应继续 scale 同一个 selected-point/waypoint recipe，而应改成显式 handle/contact pose、gripper orientation 和 post-contact temporal action decoder。

### Visible Pre-Contact Point Gate

`pre_first_close_pose_xyz` 仍然是一个 demo EE 轨迹坐标，cross-sample 偶尔能在 action imitation 上追上。因此又补了 `visible_pre_first_close_point_xyz`：先取 first close 前 `20` 帧的 demo EE xyz，再在当前 RGB-D 可见点云中找最近的真实可见 3D 点作为 label。

这个改动的意义是：

> label 本身来自当前 depth 可见几何，模型必须从 dense point tokens 中选出当前可见 pre-contact 点，不能只拟合一个抽象轨迹坐标。

Feasibility probe：

| target | normal median | cross median | EE fallback median |
|---|---:|---:|---:|
| `pre_first_close_pose` | `0.0615m` | `0.0508m` | `0.1084m` |
| `visible_pre_first_close_point` | `0.0000m` | `0.0142m` | `0.0934m` |

500-step checkpoint：

```text
runs_rlbench_visible_preclose_500/
experiments/logs/visible_preclose_500_paired/
experiments/logs/visible_preclose_500_strict/
```

selected-point 几何 gate：

| depth mode | selected point -> aux label L2 | `xyz_rmse` | `xyz_direction_cosine` |
|---|---:|---:|---:|
| normal | `0.099m` | `0.00395` | `0.561` |
| null | `0.699m` | `0.00491` | `0.251` |
| cross_sample | `0.194m` | `0.00364` | `0.722` |

paired normal-vs-cross：

| metric | value |
|---|---:|
| normal selected point -> aux label | `0.099m` |
| cross selected point -> same aux label | `0.194m` |
| selected-point normal advantage | `0.095m` |
| `paired_depth_point_xyz_l2` | `0.1746m` |
| `paired_depth_waypoint_chunk_xyz_action_l2` | `1.8790` |
| `paired_pred_xyz_l2` | `0.00818m` |

结论：

> 这轮已经证明 depth 进入了 OpenVLA 的几何 bottleneck：真实 depth 让 selected point 更接近当前可见 pre-contact label，并且该点进入 8-step waypoint/action chunk。它仍然不是 rollout success，因为 action imitation 的 cross-sample 指标更好；下一步应接 temporal/contact action decoder，而不是继续只优化 selected point。

## 3. 不能 claim 的内容

还不能说：

- OpenVLA-OFT RGB-D 已经超过 RGB-only。
- 这个 visible-object target 已经 rollout 成功。
- 这个 heuristic visible point 一定就是任务真实 contact point。

它目前只是下一轮端到端训练的代码入口和 causal diagnostic 入口。

## 4. 第一轮训练建议

先做 `open_drawer` 小门槛，不直接扩大 stable6：

```bash
DEPTH_AUX_TARGET=visible_object_point_xyz \
DEPTH_AUX_OUTPUT_DIM=3 \
DEPTH_AUX_SPATIAL_LOSS_WEIGHT=0.2 \
DEPTH_WAYPOINT_ACTION_WEIGHT=1.0 \
DEPTH_WAYPOINT_ACTION_CLIP=0.02 \
DEPTH_WAYPOINT_ACTION_SCALE=20.0 \
DEPTH_ACTION_FUSION_GATE_INIT=1.0 \
DEPTH_POINTS_PER_VIEW=1024 \
FREEZE_VLA_LORA=True \
FREEZE_PROPRIO_PROJECTOR=True \
FREEZE_ACTION_HEAD_BASE=True \
MAX_STEPS=500 \
SAVE_FREQ=500 \
BATCH_SIZE=1 \
TASKS=open_drawer \
DATASET_NAME=rlbench_open_drawer_visible_object_point \
HDF5_DIR=/root/RLBench/rgbd_hdf5_open_drawer_3demos_64 \
RUN_ROOT_DIR=/root/autodl-tmp/openvla-oft/runs_rlbench_visible_object_point \
./experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgbd
```

如果 `500` steps 的 spatial loss 和 paired geometry delta 有信号，再跑 `5000` steps。

## 5. 必须通过的 gate

训练后先跑 paired diagnostic：

```bash
RGBD_CHECKPOINT=<trained_checkpoint> \
DIAG_COMPARE_DEPTH_MODE=cross_sample \
DEPTH_AUX_OUTPUT_DIM=3 \
DEPTH_WAYPOINT_ACTION_WEIGHT=1.0 \
DEPTH_WAYPOINT_ACTION_CLIP=0.02 \
DEPTH_WAYPOINT_ACTION_SCALE=20.0 \
TASKS=open_drawer \
HDF5_DIR=/root/RLBench/rgbd_hdf5_open_drawer_3demos_64 \
DATASET_NAME=rlbench_open_drawer_visible_object_point \
./experiments/robot/rlbench/run_rlbench_rgbd_stage.sh diagnose-rgbd-normal
```

Go 条件：

- `paired_depth_point_xyz_l2` 明显大于 `0`，说明 selected point 对 normal/cross depth 敏感。
- `paired_depth_waypoint_xyz_action_l2` 明显大于 `0`，说明几何进入了 first-step action。
- `paired_pred_xyz_l2` 不再是 `1e-4` 级弱扰动，至少达到 `1e-3` 到 `1e-2` 级。
- strict normal 的 action metric 不差于 null/cross。

只有 gate 过了，才进入 closed-loop rollout。

## 6. 失败时的下一步

如果 visible-object target 仍然 no-go，不继续盲目 scale。下一步应改成更强的 object/contact target：

- 从 RLBench segmentation / mask 中提取 handle/object candidate。
- 用 projected heatmap 找到 2D contact，再 backproject 成 3D point。
- 或把 ManiSkill 的 learned cube/contact bottleneck 迁移成 OpenVLA 侧 auxiliary teacher。

核心原则不变：

> depth 必须先变成可诊断的 object/contact geometry，再进入 primary action；不能再只作为 pooled feature 或 optional residual。
