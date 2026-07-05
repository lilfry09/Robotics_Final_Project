# 下一轮 RGB-D 超过 RGB Baseline 的实验路线

更新时间：2026-07-04 UTC

## 0. 结论先行

不要继续把 clean LIBERO-Spatial 当作主战场。RGB-only OpenVLA-OFT 在 clean LIBERO 上已经接近饱和，depth 的边际收益会被天花板效应掩盖。LIBERO 后面只保留三个用途：回归测试、历史结果对照、少量 sanity check；不再用它作为主实验数据集，也不再用它支撑“RGB-D 是否超过 RGB”的核心 claim。

截至 2026-07-04，RLBench `open_drawer` 已经完成多轮 hard gate：

- safe residual
- keypose residual
- projected UV residual
- projected heatmap residual
- point-action residual
- primary waypoint action
- `farthest_future_pose_xyz` waypoint，500-step 和 5000-step

这些路线都没有让当前 OpenVLA/RLBench recipe 证明 RGB-D normal 超过 matched RGB-only，也没有让 normal depth 稳定超过 null/cross-sample。最后 5000-step `farthest_future_pose_xyz` gate 中，paired normal-vs-cross `paired_pred_xyz_l2=1.00e-04`，strict cross-sample `xyz_rmse=0.003162` 还略优于 normal `0.003167`。这个结论限制的是当前 recipe，不是证明 RGB-D/OpenVLA 方向无效。

因此下一轮目标应该更明确：

> 不再扩大当前 RLBench/OpenVLA-OFT waypoint/residual recipe。下一轮必须换成更高吞吐的数据来源和更强绑定的 3D action decoder，让 RGB-D normal 同时超过 matched RGB-only 与 null/cross-sample depth。

新的推荐路线：

1. **RLBench 保留为诊断 benchmark**：继续用它验证 action-map / decoder 的 normal/null/cross causal gate，但不再在现有 recipe 上扩大训练。
2. **ManiSkill3 作为下一轮主扩展数据源**：它官方支持 `rgbd` 和 `pointcloud` observation，并支持 GPU-parallel visual data collection，适合快速做大规模 RGB-D/point-cloud action decoder 预实验。
3. **动作空间改成 3D action decoder**：优先 DP3-style sparse point-cloud action decoder 或 Act3D/PerAct-style 3D action map；不要再把 depth 放进可选 residual。
4. **输出目标优先 object/contact-conditioned waypoint**：短步 next-pose/keypose 容易被 EE/proprio shortcut 解决，未来目标要绑定 object、handle、contact point 或 task success condition。
5. **成功门槛不放松**：RGB-D normal 必须超过 matched RGB-only，并且 normal 明显优于 null/cross-sample；只在 offline probe 上赢不算。

最新提交前后，ManiSkill3 路线已经给出正结果：

- ManiSkill3 `PushCube-v1` 和 `PickCube-v1` 官方 demo 已成功 replay 成 pointcloud + `pd_ee_delta_pos` action。
- `PushCube-v1` 20 demos 的最小 point-cloud decoder gate 为 `3/3` seeds 通过，但 paired normal-vs-cross L2 只有约 `0.0020`，更像 pipeline smoke。
- `PickCube-v1` 20 demos 的 gate 为 `2/3` seeds 通过，paired normal-vs-cross L2 mean 约 `0.0223`，比 RLBench farthest-future waypoint 的 `1e-4` 级 action delta 大得多。
- 进一步扩大到 `100` 条成功 teacher 轨迹后，raw cropped pointcloud learned policy 得到 normal `20/60`，eval-time null/cross `1/60`，matched sampled-RGB-only/null train baselines `1/60`/`3/60`。
- 因此主结论应更新为：depth/pointcloud 在 ManiSkill3 PickCube + primary pointcloud action decoder 下已经有闭环正证据；OpenVLA/RLBench no-go 只是说明旧 residual/waypoint recipe 不够。

证据见 `MANISKILL_FINAL_PILOT.md`。

硬决策：

> 从本轮开始，任何 RGB-only 已经接近天花板的 benchmark 都自动降级为 sanity check。任何让 depth 只作为 residual/side feature 的方法，除非先通过 normal/null/cross action-delta gate，否则不进入长训练或 rollout。

官方来源复核：

- ManiSkill3 文档：支持 `rgb+depth`、`pointcloud` observation，支持 replay demonstrations 并转换 observation/control modes，EE controller 支持 delta 和 non-delta control，并有 GPU-parallel RGBD/segmentation 数据采集能力。
- Act3D：把 6-DoF keypose prediction 转成 coarse-to-fine 3D action map。
- DP3：用 sparse point cloud 编码 compact 3D representation，再直接生成 action sequence。
- CALVIN：官方数据说明支持 absolute 和 relative action，用来做 action-space 对照。
- RoboTwin 2.0：官方支持 OpenVLA-OFT、DP3 等 policy，适合作为后续 benchmark 扩展。
- 完整来源核对见 `FINAL_SOURCE_AUDIT.md`。

## 0.1 方法调研后的路线修正

联网调研 PerAct、RVT、Act3D、3D Diffusion Policy、PointVLA、SpatialVLA 和 BridgeVLA 后，路线需要再收紧一点：

> 换出 LIBERO 是必要条件，但不是充分条件。真正决定 depth 能否赢 RGB 的，是 action representation 和 depth fusion 是否把 3D 几何变成“必须用”的信息。

对我们当前实验最直接的解释已经更新：

- RLBench `reach_target` RGB-only overfit 已经能成功，说明 action adapter 和 eval loop 不是完全坏的。
- 第一版 RGB-D normal 失败、RGB-D null 成功，说明当前 depth path 会破坏闭环策略，而不是提供稳定 3D correction。
- Safe RGB-anchor 版本 normal/null/cross-sample 全部 `1/1` 且 action diagnostic 完全重合，说明保护机制有效，但 depth 内容仍没有因果进入 action。
- 这和 PointVLA 的设计取向一致：先冻结/保护 vanilla action expert，只用轻量模块注入 3D，降低对强 2D/RGB policy 的扰动。
- 这也和 BridgeVLA/Act3D 的取向一致：depth 不应只是 hidden feature，而应和 keypose、heatmap 或 3D action map 对齐。

因此下一轮最小实验不再是继续 reach-only RGB-D repair，而是 **3D-sensitive causal gate**：

1. 保留 gate/clamp/frozen RGB anchor 作为不伤害 RGB policy 的默认保护。
2. 把 causal gate 从 `reach_target` 移到 `open_drawer`、`turn_tap`、`slide_block_to_target`、`pick_up_cup` 等更依赖 3D/contact 的任务。
3. keypose-conditioned residual 已经完成一次 no-go；下一步升级为 projected heatmap 或 coarse-to-fine 3D action map，让 depth 必须解决空间动作问题。
4. 评估时必须跑 normal、null、cross-sample 三种 depth。
5. 只有 normal 明显优于 null/cross-sample，才重新启动 stable6/18-task scaling。

上一阶段最可能的有效结构曾经是：

```text
RGB/OpenVLA action hidden
  -> base delta action

dense point / projected multi-view depth
  -> object/action query
  -> small gated residual or keypose-conditioned correction

final action = base delta + clamp(gate * depth_residual)
```

但 `open_drawer` 上已经连续拒绝了 safe residual、keypose residual、projected-UV residual、full heatmap residual、point-action residual 和 primary waypoint-action。最新结论是：

> 保护 RGB anchor 是必要的安全机制，但当前 OpenVLA-OFT action path 仍然没有把 metric depth 变成稳定、必要的动作信息。即使让 selected 3D point 直接成为 first-step xyz action，也只产生了很小的 normal-vs-cross action delta，并在 closed-loop 中失败。

最后尝试过的最小有效结构是：

```text
RGB/OpenVLA action hidden
  -> language/proprio/action query

dense point / 3D candidate map
  -> score/select 3D waypoint
  -> selected waypoint is the primary xyz target

final action xyz = selected waypoint - current ee xyz
final action rpy/gripper = RGB/action head or separate lightweight head
```

这条路线确实让 action 对 depth 有了微弱响应，但响应幅度不足，且 rollout 不稳定。因此它应该写成 OpenVLA/RLBench recipe 的 no-go 对照，而不是整个项目的 negative result。最终可交付主结论应放在 ManiSkill3 raw pointcloud learned policy 的 RGB-D/pointcloud gain 上。

## 1. 为什么要换数据集

当前 LIBERO clean tasks 的问题不是工程不可用，而是评测饱和：

- RGB-only 已经能在 clean trained tasks 上达到 `15/15`。
- Depth 即使有用，也很难在接近满分的 aggregate success rate 上体现。
- LIBERO-Plus 比 clean LIBERO 更好，但当前数据规模和 symbolic RGB-D 覆盖还不够。
- 继续围绕 LIBERO 调参会让问题变成“如何超过一个接近满分的 RGB baseline”，这对证明 depth 价值不公平，也不高效。

因此下一轮需要满足：

1. RGB-only baseline 不能接近满分。
2. 任务必须强依赖 3D 几何，例如精确接触、遮挡、不同高度、不同相机位姿、不同物体位置、长距离空间泛化。
3. 环境要支持 closed-loop rollout，不能只做 offline metric。
4. 能拿到 RGB-D、camera intrinsics/extrinsics、robot state 和动作。
5. 可以继续做 normal/null/shuffle depth 消融。

## 2. 数据集/评测候选

### 2.1 诊断保留：RLBench

推荐原因：

- RLBench 是成熟的大规模视觉机器人操作 benchmark，官方支持 imitation learning、multi-task learning 和 vision observations。
- 官方任务集合支持 MT15/MT30/MT55/MT100 等多任务设置。
- PerAct、RVT、Act3D 等工作都在 RLBench 上证明过 3D 表示和 6-DoF action grounding 的价值。
- 它有较强的 3D 几何需求：抽屉、杯子、peg insertion、stack blocks、turn tap、place wine 等。
- 可以做 matched RGB-only vs RGB-D，并在相同任务/变体/演示数量下比较。

注意：这一节保留 RLBench 的价值和历史路线，但不再建议扩大当前 residual/waypoint recipe。RLBench 下一步只用于验证新 point-cloud action decoder 是否比旧 recipe 更 depth-sensitive。

建议使用方式：

| 阶段 | 任务 | demos/task | 目的 |
|---|---:|---:|---|
| Pilot-0 | 6 个稳定 3D-sensitive tasks | 3 | 已完成数据转换、strict validation、revised offline probe；2000-step rollout gate 为 NO-GO |
| Pilot-0.5 | `reach_target` 单任务 | 3-10 | 先证明当前 OpenVLA-OFT action head 能在 RLBench 闭环成功 |
| Pilot-0.7 | 6 个稳定 3D-sensitive tasks | 3 | 诊断 action scale、delta-to-absolute 执行、reset 对齐和训练时长 |
| Pilot-1 | 6 个稳定 3D-sensitive tasks | 10 | 训练 matched RGB/RGB-D，验证 normal/null/shuffle rollout |
| Stage 1 | PerAct/RVT 常用 18 tasks | 10 | 看 depth 是否开始超过 RGB |
| Stage 2 | 同 18 tasks | 100 | 扩大训练量，争取稳定超过 RGB |
| Stage 3 | MT55 或 MT100 | 10-50 | 检验大规模多任务泛化 |

优先任务类型。当前默认 pilot 先用已通过 live-demo smoke 的稳定任务：

- `slide_block_to_target`
- `turn_tap`
- `close_jar`
- `open_drawer`
- `reach_target`
- `pick_up_cup`

后续扩展到更强 3D 几何任务：

- `insert_onto_square_peg`
- `stack_blocks`
- `stack_cups`
- `place_cups`
- `put_item_in_drawer`
- `turn_tap`
- `place_wine_at_rack_location`
- `put_in_safe`
- `sweep_to_dustpan`

为什么它适合证明 depth：

> RLBench 中很多任务的成功条件依赖 3D 位置、深度、高度、接触和相对空间关系；RGB-only 不容易在大变体、多任务和精确动作下刷满。

当前 RLBench 进展：

- 已生成 `stable6 x 3 demos`，共 `18 demos / 2009 transitions`。
- Dense depth -> absolute keypose offline probe 通过：normal xyz RMSE `0.0803`，null `0.6097`，cross-sample shuffle `0.2035`。
- Demo replay 在 `max_steps=200` 下 `6/6` 成功，说明 action mode 基本可执行。
- Matched policy rollout 在 `max_steps=150` 下 RGB-only 和 RGB-D normal 都是 `0/6`，主要失败是长时序漂移后触发 `InvalidActionError`。
- `reach_target` RGB-only `5000` step overfit 已通过：`MAX_DELTA_XYZ=0.03/0.05/0.08` 均为 `1/1`，成功步数均为 `29`。
- `reach_target` RGB-D dense/keypose `5000` step overfit 未通过 causal gate：
  - normal depth：`MAX_DELTA_XYZ=0.03/0.05/0.08` 全部 `0/1`。
  - null depth：`MAX_DELTA_XYZ=0.05` 为 `1/1`。
  - shuffle depth：`MAX_DELTA_XYZ=0.05` 为 `0/1`。
- `reach_target` safe RGB-anchor RGB-D 修复已完成：
  - 加载成功 RGB-only checkpoint 后再冻结 RGB/LoRA/proprio/action-head base。
  - normal/null/cross-sample rollout 全部 `1/1`，成功步数均为 `29`。
  - normal/null/cross-sample offline action diagnostic 完全相同：`xyz_rmse=0.001700`，`direction_cosine=0.97756`。
- `open_drawer` 已启动为下一轮 3D-sensitive gate：
  - HDF5 子集：`3 demos / 317 transitions`。
  - dense-depth keypose probe 通过：normal xyz RMSE `0.0705`，null `0.1420`，shuffle `0.1466`。
  - matched RGB-only 单任务 baseline 已训练 `5000` step。
  - RGB-only eval：`0/1`，horizon `200`，length `200`，无 `InvalidActionError`。
  - safe RGB-D eval：normal/null/cross-sample 全部 `0/1`，length `200`。
  - safe RGB-D action diagnostic：normal 和 cross-sample 完全相同，`xyz RMSE=0.001336`，cosine `0.78733`；null 只略差。
  - keypose-conditioned residual 已训练 `5000` step，但仍未通过因果门：paired normal-vs-cross-sample action delta 全部为 `0.0`；normal RMSE `0.0137849`，null `0.0137148`，cross-sample `0.0137849`。

因此当前不是回头刷 LIBERO，也不是马上扩大 stable6。最小闭环结论是：

> RLBench action adapter 可用，因为 RGB-only reach 能成功；safe RGB-anchor fusion 也可用，因为 RGB-D 不再破坏 reach。但 reach 太容易，normal/null/cross-sample 完全一致，不能证明 depth 内容有效。下一步必须换到更 3D-sensitive 的任务和更显式的 spatial action target，再考虑 stable6 或 18-task scale。

`open_drawer` 的新增结论是：

> 数据集更换确实有必要，但还不够。即使在 RGB-only 不饱和、depth offline keypose / projected heatmap probe 通过的任务上，shallow safe residual、keypose-conditioned residual、projected UV/full heatmap residual、point-action residual 和 primary waypoint-action 都没有把真实 depth 内容转化成稳定闭环收益。最终报告应把这作为当前 OpenVLA/RLBench recipe 的系统性 no-go，对照 ManiSkill3 primary pointcloud decoder 的正结果来讲。

### 2.1.1 已完成 no-go：`open_drawer` keypose residual gate

当前不要回到 LIBERO，也不要马上扩大 stable6。我们已经完成了一个小而硬的 gate：

> 在 `open_drawer` 上，让 absolute keypose prediction 参与 final delta action；normal depth 必须先在 offline diagnostic 中和 cross-sample depth 拉开，再进入 closed-loop rollout。

为什么这样改：

- 之前的 absolute keypose 只是 aux loss，policy 可以预测 keypose 但不把它用于 action。
- safe RGB-anchor 保护了 RGB policy，但 normal/null/cross-sample 动作几乎一样，说明 depth 没有进入 action causal path。
- keypose-conditioned residual 把 depth branch 的空间输出接回 action chunk，贴近 PerAct/Act3D 的 keypose/action grounding 思路，同时仍保留 OpenVLA-OFT 的 delta control。

已实现的代码开关：

```text
--depth_keypose_residual_weight
--depth_keypose_residual_clip
```

实际训练命令：

```bash
HDF5_DIR=/root/RLBench/rgbd_hdf5_open_drawer_3demos_64 \
DATASET_NAME=rlbench_open_drawer_3demos_64 \
RUN_ROOT_DIR=/root/runs_rlbench_open_drawer_keypose_residual \
TASKS=open_drawer \
MAX_STEPS=5000 \
SAVE_FREQ=1000 \
DEPTH_POINTS_PER_VIEW=1024 \
DEPTH_ACTION_FUSION_GATE_INIT=1.0 \
DEPTH_HIDDEN_DELTA_CLIP=0.001 \
DEPTH_ACTION_RESIDUAL_CLIP=0.02 \
DEPTH_KEYPOSE_RESIDUAL_WEIGHT=1.0 \
DEPTH_KEYPOSE_RESIDUAL_CLIP=0.02 \
DEPTH_AUX_SPATIAL_LOSS_WEIGHT=0.2 \
DEPTH_DROPOUT=0.2 \
FREEZE_VLA_LORA=True \
FREEZE_PROPRIO_PROJECTOR=True \
FREEZE_ACTION_HEAD_BASE=True \
RESUME_COMPONENTS_FROM=/root/runs_rlbench_open_drawer_3demos/47a0ec7fc4ec123775a391911046cf33cf9ed83f+rlbench_open_drawer_3demos_64+rgb-only+b1+lr-0.0001+lora-r4+dropout-0.0--rlbench-rgb-only \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgbd
```

训练后先跑的 gate：

```bash
RGBD_CHECKPOINT=<new-keypose-residual-checkpoint> \
HDF5_DIR=/root/RLBench/rgbd_hdf5_open_drawer_3demos_64 \
DATASET_NAME=rlbench_open_drawer_3demos_64 \
TASKS=open_drawer \
DEPTH_KEYPOSE_RESIDUAL_WEIGHT=1.0 \
DEPTH_KEYPOSE_RESIDUAL_CLIP=0.02 \
DEPTH_HIDDEN_DELTA_CLIP=0.001 \
DEPTH_ACTION_RESIDUAL_CLIP=0.02 \
DIAG_COMPARE_DEPTH_MODE=cross_sample \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh diagnose-rgbd-all-strict
```

结果：

- 训练稳定完成到 `5000` steps。
- Paired normal vs cross-sample diagnostic：`paired_pred_l1=0.0`，`paired_pred_rmse=0.0`，`paired_pred_xyz_l2=0.0`。
- Strict diagnostic：normal RMSE `0.0137849`，null RMSE `0.0137148`，cross-sample RMSE `0.0137849`。
- 判定：NO-GO。normal depth 没有让 action prediction 对 depth 内容产生可测差异，因此不跑长闭环，也不扩大 stable6。

### 2.1.2 已完成 no-go：projected heatmap / point-action residual

keypose residual 之后又完成了三条更强的 gate：

1. `projected_keypose_uv`
   - 输出两路相机 normalized UV，共 `4` 维。
   - 训练稳定到 `5000` steps。
   - paired normal-vs-cross action delta 仍为 `0.0`。

2. `projected_keypose_heatmap`
   - 输出两路相机 `16x16` Gaussian heatmap，共 `512` 维。
   - action residual 接收 full logits 和 soft-argmax UV。
   - 训练稳定到 `5000` steps。
   - paired normal-vs-cross action delta 仍为 `0.0`。

3. `point_keypose_xyz`
   - action/language query score dense 3D points。
   - soft selected point 预测 keypose xyz，并接 bounded translation residual。
   - 修复了 RLBench negative focal-length backprojection bug。
   - 训练稳定到 `5000` steps。
   - paired normal-vs-cross action delta 仍为 `0.0`。

判定：

> Projected heatmap 和 point-keypose 都能作为 auxiliary/offline signal，但只要通过 bounded residual 接回 RGB action path，final action 仍然可以完全忽略 depth。

### 2.1.3 已完成 no-go：primary 3D waypoint action

最后一次尝试不是继续调 residual clip、heatmap size、point residual weight，而是把 depth 输出改成主动作表示：

1. **Primary 3D waypoint head**
   - 从 dense point tokens 或 workspace candidates 中选择下一 3D waypoint。
   - `xyz` action 直接来自 `selected_waypoint - current_ee_xyz`。
   - RGB base action 只预测 rpy/gripper，或作为 ablation fallback。
   - paired normal-vs-cross-sample 必须改变 selected waypoint 和 final xyz action。

2. **Act3D/PerAct-style 3D action map**
   - 在 workspace 里预测 coarse-to-fine gripper keypose。
   - 用 predicted 3D keypose 直接形成 delta translation。
   - 评估 normal/null/cross-sample 同一样本 action delta。

3. **DP3-style 3D action decoder**
   - point cloud encoder 输出 compact 3D token。
   - action decoder / diffusion head 直接生成 delta action chunk。
   - normal/null/cross-sample gate 放在生成动作差异上。

保留为辅助诊断，但不再作为主路线：

1. **BridgeVLA-style projected heatmap**
   - 从 depth/point cloud 投影到多视角 2D plane。
   - 预测 gripper/keypoint heatmap 或 object-contact heatmap。
   - 只能证明 spatial target 有 signal，不能单独证明 policy 使用 depth。

2. **PointACT-style multi-scale point-action attention**
   - 让 action/query tokens 直接 attend 多尺度 point tokens。
   - 避免单个 pooled query 把 local geometry 抹平。
   - 只有当 action tokens 的输出直接成为 final xyz action，才值得继续。

实际结果：

| check | result |
|---|---:|
| paired `normal` vs `cross_sample` `paired_pred_l1` | `5.90e-05` |
| paired `normal` vs `cross_sample` `paired_pred_rmse` | `1.15e-04` |
| paired `normal` vs `cross_sample` `paired_pred_xyz_l2` | `3.05e-04` |
| strict normal `xyz_rmse` | `0.003178` |
| strict null `xyz_rmse` | `0.003210` |
| strict cross-sample `xyz_rmse` | `0.003226` |
| rollout normal | `0/1`, length `11`, `InvalidActionError` |
| rollout null | `0/1`, length `10`, `InvalidActionError` |
| rollout cross-sample | `0/1`, length `11`, `InvalidActionError` |

判定：

> NO-GO。Primary waypoint-action 不再是完全 0 causal delta，但差距太小，且 closed-loop normal/null/cross 全失败，不能作为 RGB-D 有效的证据。

当前实现更新：

- 已新增 RLBench projected-keypose heatmap offline gate。
- `open_drawer` gate 通过：normal peak error `2.85px`，cross-sample `8.86px`，null `43.30px`。
- stable6 gate 通过：normal peak error `2.94px`，cross-sample `12.53px`，null `57.63px`。
- 已拒绝 `projected_keypose_uv`、`projected_keypose_heatmap`、`point_keypose_xyz` residual gate 和 primary waypoint-action gate。
- 由于明天提交，当前不再建议启动新训练；最终材料应强调完整 pipeline、严格 causal ablation、LIBERO 饱和判断、RLBench `open_drawer` 最终 no-go，以及下一步需要更彻底的 3D action-space redesign。

### 2.2 第二优先：ManiSkill

推荐原因：

- ManiSkill 3 强调 GPU parallel simulation 和高速 RGBD + segmentation 数据采集。
- 适合快速扩大训练量，做大量 object pose / viewpoint / distractor 随机化。
- 可以很方便地产生 depth、segmentation、state、成功指标和可控扰动。

适合用途：

- 快速生成大规模 RGB-D 数据。
- 做 depth branch 的 offline/pretraining。
- 做受控的 camera/viewpoint/height perturbation 评测。

局限：

- 和当前 OpenVLA-OFT/LIBERO 管线差异较大，需要新 adapter。
- 语言任务复杂度不一定天然接近 VLA benchmark，需要筛选任务或补 language template。

建议：

> 如果 RLBench 接入成本太高或生成数据太慢，ManiSkill 是最适合做大规模 synthetic RGB-D scaling 的第二选择。

### 2.3 第三优先：RoboCasa / RoboCasa365

推荐原因：

- RoboCasa365 有 365 个任务、2500+ kitchen scenes，以及 2200+ hours demonstrations。
- 更接近 household manipulation 和 generalist robot 场景。
- 适合在更大规模、更复杂场景中比较 RGB-only 和 RGB-D。

局限：

- 工程接入成本更高。
- 任务和资产规模大，第一轮不适合直接全量上。
- 需要先确认 depth/camera parameter 导出和动作格式能否稳定适配。

建议：

> RoboCasa 更适合作为第二阶段大规模泛化验证，不作为第一轮证明 depth causal gain 的最小实验。

### 2.4 CALVIN

优点：

- 官方支持 RGB、depth、proprioception。
- 官方明确支持 absolute cartesian pose、relative cartesian displacement、joint action 三种 action spaces。
- 长时序语言任务比 clean LIBERO 更难。

局限：

- 主要价值在 long-horizon language-conditioned control，不一定最直接体现 metric depth 的几何优势。
- 如果目标是先证明 depth > RGB，RLBench 更直接。

建议：

> CALVIN 更适合做 action space 对照，尤其是 absolute vs relative cartesian action；但不是第一优先主 benchmark。

### 2.5 DROID / BridgeData V2 / Open X-Embodiment

定位：

- 这些是真实机器人和大规模预训练数据，不适合作为第一轮 closed-loop 因果评测主场。
- 更适合做 representation pretraining、depth adapter pretraining、language/RGB anchor pretraining。

可用价值：

- DROID：76k demonstrations，350h interaction，包含多场景、多视角和 camera calibration。
- BridgeData V2：60,096 trajectories，包含多环境、多技能和部分 depth 视角。
- Open X-Embodiment：1M+ real robot trajectories，22 robot embodiments，大规模 action-language 数据。

限制：

- closed-loop eval 不如模拟环境直接。
- action spaces、camera setups、depth availability 不统一。
- 如果直接混训练，容易把因果问题变成 dataset mixture 问题。

建议：

> 先在 RLBench/ManiSkill 证明 RGB-D causal gain，再考虑用真实大数据做预训练或泛化增强。

## 3. Action Space 重新设计

当前本地代码的基本接口：

- `ACTION_DIM = 7`
- `NUM_ACTIONS_CHUNK = 8`
- 连续动作头输出 action chunk
- 当前 OpenVLA 语义接近 end-effector delta / normalized continuous action

下一轮不要只问“depth 怎么融合”，还要问：

> 什么 action target 最能让 depth 的 3D 几何优势进入控制？

### 3.1 三种 action target 必须对照

| 方案 | 输出 | 优点 | 风险 |
|---|---|---|---|
| Delta chunk | 未来 `H x 7` 相对动作 | 学习稳定，和当前代码兼容 | depth 只提供局部修正，容易被 RGB/proprio 绕过 |
| Absolute keypose | base/world frame 下的下一关键位姿 | 和 metric depth 直接对齐 | 学习分布更宽，控制要靠 planner/IK |
| Hybrid | absolute keypose/heatmap + delta residual | 兼顾 3D grounding 和闭环稳定 | 实现复杂，但最值得做 |

推荐主线：

> 保留 delta action chunk 作为执行接口，同时增加 absolute keypose/heatmap 辅助头。训练时让 depth 预测 3D keypose，执行时把 keypose 转成 delta residual 或 planner target。

### 3.2 为什么 absolute keypose 对 depth 重要

Depth 的优势是 metric geometry：物体、目标、机械臂、相机之间的真实 3D 关系。

如果输出只是短 horizon delta action，模型可能继续靠 RGB/proprio 学局部动作模式；depth 只变成小扰动。

如果加入 absolute keypose 或 3D action map：

- depth tokens 必须定位 object/target/contact。
- action target 和 3D 表示处于同一个 base/world frame。
- normal/null/shuffle 的差异更容易反映到 loss 和 rollout。

### 3.3 不是简单放弃 delta

Action-space 相关研究提示：delta actions 通常更好学、更稳定；absolute actions有全局 grounding，但学习难度更高。

所以推荐不是“全改 absolute”，而是：

```text
主执行头：delta action chunk
空间辅助头：absolute 3D keypose / heatmap / action map
融合方式：absolute head 提供 waypoint 或 residual context
```

这样既不丢掉现有 OpenVLA-OFT 的稳定性，也给 depth 一个必须解决的 3D 子问题。

## 4. Depth Fusion 重新设计

旧方法的问题：

- grid-pooled depth summary 丢失 spatial structure。
- action-side summary 太容易被 action head 忽略。
- prefix append 容易扰动 VLM。
- 只做 contact distance / global aux 不足以强制使用 depth。

下一轮推荐改成三层结构。

### 4.1 Dense 3D tokens

从 RGB-D 和 K/T 得到 point cloud 或 3D feature cloud：

```text
depth + K + T
  -> base-frame point cloud
  -> sample 1024-4096 points
  -> features: xyz_base, z_cam, RGB/vision feature, view_id, valid, optional segmentation
  -> 3D positional encoding
```

不要先池化成 `4x4` 或 `8x8` summary。可以后面由 cross-attention 自己聚合。

当前仓库已新增第一版 dense token encoder：

```text
prismatic/models/dense_point_depth_encoder.py
```

训练脚本已支持：

```text
--depth_encoder_type dense_point
--depth_num_points_per_view 1024
```

### 4.2 Object/action query cross-attention

用语言和 action hidden state 形成 query，去 attend 3D tokens：

```text
query = f(language tokens, action hidden, proprio)
context = CrossAttention(query, dense_3d_tokens)
```

这比 global pooling 更合理，因为同一张 depth map 中不是所有几何都相关，模型必须查询“当前任务关心的 object/target/contact region”。

### 4.3 3D action map / heatmap head

增加一个显式空间预测头：

```text
dense_3d_tokens
  -> 3D action map 或 2D projected heatmap
  -> predict next keypose / object target / contact point
```

对应监督：

- RLBench：next keypose / gripper pose / task success-relevant waypoint。
- ManiSkill：object pose、target pose、end-effector target、contact point。
- LIBERO-Plus：`ee_to_object_xyz`、`object_to_target_xyz`、`gripper_to_contact_distance`。

当前仓库已支持 RLBench absolute keypose auxiliary：

```text
--aux_target absolute_keypose
--aux_output_dim 8
```

训练前新增 GO/NO-GO probe：

```text
experiments/robot/rlbench/probe_dense_depth_keypose.py
```

它训练一个小型 dense-depth query head 预测 `rlbench_keypose_action`，然后对同一个模型评估：

- normal depth
- null depth
- shuffled geometry

只有 normal 的 keypose xyz RMSE 明显低于 null/shuffle，才继续扩大 VLA 训练。

### 4.4 训练 loss

建议组合：

```text
L = L_delta_action
  + lambda_keypose * L_absolute_keypose
  + lambda_heatmap * L_spatial_heatmap
  + lambda_aux * L_task_3d
  + lambda_contrast * L_normal_vs_corrupt
```

其中 `L_normal_vs_corrupt` 不一定直接让 corrupt loss 变差，但至少要监控：

- normal action loss < null action loss
- normal spatial loss < shuffle spatial loss
- normal rollout success > null/shuffle rollout success

## 5. 训练量扩大方案

### 5.1 不再用 5 tasks x 20 demos

旧训练规模太小：

- 5 tasks
- 89 demos
- 约 11k transitions

这个规模只能验证 pipeline，不能证明 robust depth gain。

### 5.2 新训练规模策略

旧版路线是继续扩大 RLBench，但当前已经有 `open_drawer` 多轮 no-go，因此不建议把算力继续投到同一 OpenVLA-OFT residual/waypoint recipe。新的训练规模应该分两条线：

| 阶段 | 数据 | 训练步数建议 | 目标 |
|---|---:|---:|---|
| Adapter smoke | ManiSkill3 1-2 tasks x small demos | 1k-5k | 跑通 RGB-D/pointcloud 导出、language template、action normalization |
| Decoder pilot | ManiSkill3 3-5 contact/pose tasks x 50-100 demos | 20k-80k | 训练 point-cloud action decoder，看 normal/null/cross action delta |
| RLBench regression | `open_drawer` / stable6 x 3 demos | 2k-10k | 验证新 decoder 是否比旧 waypoint recipe 更依赖 depth |
| Main pilot | ManiSkill3 或 RLBench 6-10 tasks x 100 demos | 100k-300k | matched RGB-only vs RGB-D normal/null/cross |
| Scale | ManiSkill3 large synthetic 或 RoboTwin/RoboCasa pilot | 300k-1M+ | 多任务泛化和 benchmark transfer |

当前本机状态：

```text
ManiSkill3 已安装到独立 venv：/root/autodl-tmp/envs/maniskill3-venv
PushCube-v1 state smoke 已通过。
PushCube-v1 pointcloud smoke 已通过。
HDF5 validation 已通过。
```

所以第一步已经从“安装环境”推进到“做小规模可复现数据导出”。下一步不是直接大训练，而是：

1. 选 1-2 个强 3D/contact 任务做 RGB-D/pointcloud 数据导出。
2. 转成与当前 HDF5 loader 兼容的最小 schema。
3. 训练一个小 point-cloud action decoder，而不是继续接 OpenVLA-OFT residual。

每个阶段都必须训练 matched pairs：

1. RGB-only。
2. RGB-D normal。
3. RGB-D eval with null depth。
4. RGB-D eval with shuffle/corrupt depth。

只训练 RGB-D 没意义，必须有 matched RGB-only baseline。

## 6. 成功标准

### 6.1 Rollout gate

第一阶段不要追求漂亮大表，先看硬门槛：

```text
RGB-D normal >= RGB-only + 5 percentage points
RGB-D normal >= RGB-D null + 5 percentage points
RGB-D normal >= RGB-D shuffle + 5 percentage points
```

如果 task 数少，用 absolute tasks count 表示：

```text
normal 比 RGB-only 至少多成功 3/60 trials
normal 比 null 至少多成功 3/60 trials
normal 比 shuffle 至少多成功 3/60 trials
```

### 6.2 Spatial gate

在 rollout 之前先要求 offline spatial probe 通过：

```text
normal keypose error << null/shuffle keypose error
normal heatmap PSNR > null/shuffle by clear margin
normal 3D point error lower by at least 30%
```

如果 offline spatial gate 都不过，不要跑大训练。

### 6.3 Category gate

提升应该主要出现在：

- camera/viewpoint variation
- object position variation
- target height / shelf / drawer level variation
- insertion / stacking / contact-rich tasks

如果提升只出现在 easy pick-place，说明还没真正体现 depth 价值。

## 7. 下一轮最小可行实验

### 7.1 任务选择

RLBench 保留为 regression，不再作为下一轮正结果主扩展。下一轮优先从 ManiSkill3 选任务，标准如下：

```text
必须有明显 3D/contact 需求；
RGB-only 不应轻易刷满；
能导出 RGB-D 或 pointcloud；
能做 normal/null/cross-sample depth 消融；
动作空间能转成 absolute pose、delta pose 或 action chunk。
```

如果 ManiSkill3 adapter 尚未完成，临时 regression 仍可使用 RLBench 已有任务：

1. `slide_block_to_target`
2. `turn_tap`
3. `close_jar`
4. `open_drawer`
5. `reach_target`
6. `pick_up_cup`

每个任务：

- ManiSkill3 pilot：优先 `50-100 demos/task`，因为 synthetic 数据吞吐更高。
- RLBench regression：只做小规模 gate，不再扩大旧 recipe。
- eval 每个任务至少 `20-25` episodes；如果时间有限，先用 `5` episodes 做 smoke，再扩到 `25`。

### 7.2 模型对照

训练三条线，但第三条必须换 action decoder：

1. RGB-only OpenVLA-OFT/OFT-style action head。
2. RGB-D dense 3D tokens + OpenVLA-OFT delta action chunk，作为历史对照。
3. RGB-D point-cloud action decoder，例如 DP3-style compact 3D encoder + action chunk decoder，或 Act3D/PerAct-style 3D action map。

评估四种输入：

1. RGB-only baseline。
2. RGB-D normal。
3. RGB-D null depth。
4. RGB-D shuffle/corrupt depth。

### 7.3 决策规则

继续扩大到 `18 tasks x 100 demos` 的条件：

- RGB-D normal 超过 RGB-only。
- normal 明显超过 null/shuffle。
- 提升集中在 spatial/contact tasks。
- offline keypose/heatmap/action-map probe 明显通过。
- paired normal-vs-cross action delta 不再是 `1e-4` 级弱扰动，而是能改变 selected action target 或 action chunk。

停止或重设计的条件：

- normal 约等于 null：depth 仍被忽略。
- shuffle 大于等于 normal：模型学到非因果 artifacts。
- normal 超过 null/shuffle 但低于 RGB-only：depth 有信号但融合破坏 policy，需要减小 residual/gate 或加强 RGB anchor freeze。
- reach-only 全部成功但 normal/null/cross-sample 完全相同：只能算 sanity pass，不能进入 depth claim。

## 8. 需要改的代码模块

最小代码改动清单：

1. ManiSkill3 数据 adapter：
   - 新增 `experiments/robot/maniskill/`。
   - 导出统一 HDF5：RGB、depth/pointcloud、camera params、proprio、actions、language、success metadata。
   - 如果先不接 VLA，至少先导出 pointcloud/action chunks 供 decoder 训练。
2. 新数据 adapter 保留 RLBench regression：
   - `experiments/robot/rlbench/`
   - 生成统一 HDF5：RGB、depth、K、T、proprio、actions、language、keypose labels。
3. Dense depth encoder：
   - 不再只输出 grid-pooled tokens。
   - 支持 point sampling、3D positional encoding、view embedding。
4. Action decoder：
   - 保留原 7D delta chunk。
   - 新增 point-cloud action decoder 或 3D action-map head。
   - 不再把 depth 只作为 bounded residual 接回 RGB action path。
5. Eval runner：
   - 支持 RLBench rollout。
   - 支持 ManiSkill3 rollout 或至少 policy-vs-demo diagnostic。
   - 支持 normal/null/shuffle depth。
6. Offline probe：
   - keypose error。
   - heatmap/3D action map error。
   - normal vs null/shuffle separation。

## 9. 参考来源

数据集/环境：

- RLBench GitHub: https://github.com/stepjam/RLBench
- ManiSkill docs: https://maniskill.readthedocs.io/en/latest/
- RoboCasa GitHub: https://github.com/robocasa/robocasa
- CALVIN GitHub: https://github.com/mees/calvin
- DROID project: https://droid-dataset.github.io/
- BridgeData V2: https://rail-berkeley.github.io/bridgedata/
- Open X-Embodiment: https://robotics-transformer-x.github.io/

3D/depth/action representation：

- PerAct: https://peract.github.io/
- RVT: https://robotic-view-transformer.github.io/
- Act3D: https://act3d.github.io/
- 3D Diffusion Policy: https://3d-diffusion-policy.github.io/
- PointVLA: https://pointvla.github.io/
- SpatialVLA: https://spatialvla.github.io/
- BridgeVLA: https://bridgevla.github.io/
- Action space design: https://arxiv.org/html/2602.23408v1

## 10. 下一步执行顺序

1. 停止当前 RLBench waypoint/residual recipe。
   - 证据：`farthest_future_pose_xyz` 5000-step gate 未过，normal 没有超过 cross-sample。
   - 动作：不再 rollout，不再扩大 stable6/18-task 同 recipe。

2. 建 ManiSkill3 adapter smoke。
   - 目标：导出 1-2 个任务的 RGB-D/pointcloud/proprio/action/language 到统一 HDF5。
   - 当前本机未安装 ManiSkill3，所以第一步是环境安装和最小采集脚本。
   - 输出：`experiments/robot/maniskill/`、一个 HDF5 样例、一个 validation script。

3. 做 point-cloud action decoder pilot。
   - 不接 OpenVLA-OFT residual。
   - 先训轻量 DP3-style decoder：point cloud encoder + proprio/language embedding + action chunk output。
   - gate：same observation 下 normal vs null/cross 必须改变 action prediction，并且 normal action loss 低于 null/cross。

4. 用 RLBench `open_drawer` 做 regression。
   - 把新 decoder 或 action-map head 回测到已有 RLBench HDF5。
   - 目标不是刷成功率，而是确认新 action decoder 是否比旧 waypoint recipe 更 depth-sensitive。

5. 过 offline/action-delta gate 后再扩大。
   - ManiSkill3 pilot：`3-5 tasks x 50-100 demos`。
   - RLBench regression：`open_drawer` + stable6 小规模。
   - 只有 normal 同时超过 RGB-only 和 null/cross 后，才考虑 RoboTwin/RoboCasa 或 18-task RLBench。

核心原则：

> LIBERO 已经饱和，不能再作为主战场；RLBench 已经证明当前 residual/waypoint recipe 不够。下一轮不是单纯换 benchmark，而是把数据吞吐、point-cloud representation 和 action decoder 一起换掉。
