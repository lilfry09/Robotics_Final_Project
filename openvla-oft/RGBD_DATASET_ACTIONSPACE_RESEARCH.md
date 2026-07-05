# RGB-D 数据集、Action Space 与 Depth Fusion 调研笔记

更新时间：2026-07-04 UTC

## 0. 结论

下一轮主线不应该继续在 clean LIBERO 上证明 depth，因为 RGB-only 已经接近天花板。LIBERO 现在只适合作为历史对照、pipeline sanity check 和少量回归测试，不再作为“RGB-D 是否超过 RGB”的主证据。更合理的目标是：

> 在 RGB-only 尚未饱和、任务强依赖 3D 几何、并支持 closed-loop rollout 的 benchmark 上，验证 RGB-D normal 是否稳定超过 matched RGB-only，同时显著超过 null/shuffle depth。

当前推荐排序：

1. **ManiSkill3**：下一轮正结果第一优先，适合快速扩大 RGB-D/point cloud 数据量和做高吞吐 action-map / action decoder 预实验。
2. **RLBench**：保留为 regression/diagnostic benchmark，因为本仓库环境和转换/eval 已经打通，但当前 OpenVLA-OFT residual/waypoint recipe 已经多轮 no-go，不再扩大同一路线。
3. **RoboTwin 2.0**：候选第三优先，适合后续双臂/强扰动/合成数据路线，且官方已有 OpenVLA-OFT/DP3 等 policy 支持。
4. **RoboCasa/RoboCasa365**：适合第二阶段 household 泛化验证，规模大但接入成本高。
5. **DROID / Open X-Embodiment**：适合作为真实数据预训练或 representation 学习来源，不适合作为第一轮 closed-loop causal gate。
6. **CALVIN**：适合做 absolute vs relative action-space 对照，但不是第一轮证明 3D 几何收益的主战场。

当前阶段的判断：

> 不要再优化 LIBERO 上的成功率曲线，也不要继续扩大当前 RLBench waypoint/residual recipe。RLBench 这轮只证明当前 action path 没有把 depth 变成必要 action 信息；它不证明 OpenVLA/RGB-D 没用。下一轮应优先做 ManiSkill3 adapter + primary point-cloud action decoder。

提交前的最新结果进一步支持这个判断：

- ManiSkill3 `PushCube-v1` / `PickCube-v1` 官方 demo 已 replay 成 pointcloud + `pd_ee_delta_pos` action。
- 小型 PointNet action decoder 在 `PickCube-v1` 上得到约 `0.0223` 的 paired normal-vs-cross action L2 mean，明显高于 RLBench final waypoint recipe 的 `1e-4` 量级。
- 继续扩大到 `100` 条成功 teacher 轨迹后，raw cropped pointcloud learned policy 达到 normal `20/60`，eval-time null/cross `1/60`，matched sampled-RGB-only/null train baselines `1/60`/`3/60`。
- 因此现在可以 claim 的不是 “OpenVLA 端到端 RGB-D 已成功”，而是 “ManiSkill3 primary pointcloud action decoder 证明 depth/pointcloud 几何可以带来闭环收益”。

详细证据：`MANISKILL_FINAL_PILOT.md`。

项目级约束：

> LIBERO 不再是主数据集。后续任何新方法如果只在 clean LIBERO 上提升，最多算 pipeline smoke 或历史对照；必须在 ManiSkill3/RLBench 这类未饱和、3D-sensitive 的 closed-loop benchmark 上通过 matched RGB-only 与 normal/null/shuffle depth 对照，才算有效进展。

## 0.1 联网调研依据

本轮只采用官方仓库、项目主页和论文项目页作为依据，避免用二手博客做实验路线判断。

| 来源 | 关键信息 | 对本项目的决策影响 |
|---|---|---|
| RLBench 官方仓库 | 定位为大规模 vision-guided manipulation benchmark，覆盖 imitation learning、multi-task learning、geometric computer vision 等方向 | 作为 regression/diagnostic benchmark，替代已饱和的 clean LIBERO 做硬门控 |
| ManiSkill3 官方文档 | 支持 `rgb+depth`、`rgb+depth+segmentation`、`pointcloud` observation；trajectory replay 可转换 observation/control modes；EE controllers 支持 delta 和 non-delta control | 已从候选升级为最终正结果主线：PickCube raw pointcloud policy normal `20/60`，matched no-depth baselines `1/60` / `3/60` |
| CALVIN 官方仓库 | 支持 RGB-D sensors，并明确提供 absolute cartesian pose、relative cartesian displacement、joint action 三类 action space | 用来参考 action-space 对照，尤其 absolute vs relative，不作为第一主 benchmark |
| RoboCasa/RoboCasa365 官网 | RoboCasa365 覆盖 365 tasks、2500 kitchen environments，并有大量 human/synthetic demonstrations | 适合后期 household 泛化和规模验证，第一轮接入成本偏高 |
| RoboTwin 2.0 官方仓库 | 支持数据采集、leaderboard 和多种 policy baseline，包括 DP3、RDT、PI0、OpenVLA-OFT | 适合后续合成数据与强 domain randomization 路线，但双臂/环境接入成本高于当前 RLBench gate |
| DROID / Open X-Embodiment 项目页 | DROID 有 76k trajectories / 350h / 564 scenes / 86 tasks；Open X-Embodiment 有 1M+ trajectories / 22 embodiments | 适合真实数据预训练，不适合作为第一轮 normal/null/cross depth closed-loop 因果评测 |
| PerAct 项目页 | 强调 RGB-D voxel observation 与 discretized 6-DoF action 的统一空间 | 不继续做浅层 depth append，而是引入 absolute keypose / 3D action map 辅助 |
| RVT 项目页 | 主题是 Robotic View Transformer for 3D Object Manipulation | 强化 multi-view / re-rendering / spatial action representation 的方向 |
| Act3D 项目页 | 主题是 3D feature field transformers for multi-task robotic manipulation | 支持用 3D feature field / query-style action detection 替代全局 pooled depth summary |
| DP3 项目页 | sparse point cloud + compact 3D encoder + diffusion action generator | 下一步若继续提升，应把 raw pointcloud/object bottleneck 接到 diffusion/temporal action decoder |
| PointVLA / SpatialVLA 项目页 | point cloud 注入 VLA、Ego3D position encoding、adaptive spatial action grids | 支持保护 RGB anchor，同时把 3D 信息放进 robot-centric spatial/action token |

参考链接：

- RLBench: https://github.com/stepjam/RLBench
- ManiSkill3 observation: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html
- ManiSkill3 replay: https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html
- ManiSkill3 action spaces: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html
- CALVIN: https://github.com/mees/calvin
- RoboCasa: https://robocasa.ai/
- RoboCasa repo: https://github.com/robocasa/robocasa
- RoboTwin 2.0: https://github.com/robotwin-Platform/robotwin
- DROID: https://droid-dataset.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- PerAct: https://peract.github.io/
- RVT: https://robotic-view-transformer.github.io/
- Act3D: https://act3d.github.io/
- DP3: https://3d-diffusion-policy.github.io/
- PointVLA: https://pointvla.github.io/
- SpatialVLA: https://spatialvla.github.io/

## 0.2 2026-07-03 方法搜索后的新增判断

这轮联网搜索后，最重要的结论不是“换一个更大的数据集就会好”，而是：

> 对于 RGB-D/VLA，depth 必须进入 action 表示和训练目标；如果只是作为可选 feature 接到 action head，强 RGB/proprio policy 很容易绕开它，甚至被它扰乱。

几个方法给出的共同信号：

| 方法 | 关键设计 | 对我们当前失败的启发 |
|---|---|---|
| PerAct | RGB-D voxel observation 和离散 6-DoF next-best voxel action 绑定 | depth 的优势来自 3D observation-action 同空间，不是附加 token |
| RVT / RVT-2 | 多视角 re-rendering / view transformer，预测 gripper pose | 3D 表示可以先投到更适合 transformer 的多视角结构，而不是直接全局池化 |
| Act3D | 把 6-DoF keypose prediction 当作 3D detection，coarse-to-fine 采样 3D action map | 我们应该做 3D action map/keypose query，而不是只做 pooled depth summary |
| 3D Diffusion Policy / DP3 | sparse point cloud + compact 3D encoder + action generative model | 点云不需要极重，但必须保留空间结构，并让动作模型直接条件化在 3D 表示上 |
| PointVLA | 冻结 vanilla action expert，用轻量模块注入 point cloud，并尽量减少对原 2D policy 的扰动 | 贴合我们当前 reach 结果：RGB-only 能成功，RGB-D normal 失败，所以先保护 RGB anchor |
| SpatialVLA | Ego3D position encoding 和 adaptive spatial action grids | depth/3D 信息最好被编码成 robot-centric spatial/action tokens |
| BridgeVLA | 3D point cloud 投影成多视角 2D image，输出 2D heatmap，做 input-output alignment | 解释了 heatmap 为什么该继续做，但不能用会丢 spatial structure 的 grid pooling |

因此，下一步不应该继续尝试：

- 在 LIBERO 上调 depth token 或 checkpoint interpolation。
- 简单扩大当前 dense point + object-query recipe。
- 让 depth 分支以大 gate 直接影响 action head。

下一步应该改成：

1. **先保护 RGB anchor**：冻结 VLA LoRA、proprio projector 和 action-head base，只训练 depth/query/residual。
2. **小门控残差接入**：depth residual gate 从 `0.01` 或更低开始，并使用 `depth_hidden_delta_clip` / `depth_action_residual_clip` 保护动作路径，避免 normal depth 把已成功的 RGB policy 拉偏。
3. **BridgeVLA-style spatial target**：不要再用全局 pooled heatmap；改成 point-cloud multi-view projection 或 dense 2D/3D action map。
4. **PointVLA-style minimal injection**：优先找 action expert 中“可以替换/注入但不破坏原策略”的位置，而不是到处加 depth。
5. **强 causal gate**：normal 必须同时赢 RGB-only、null 和 cross-sample depth；只赢 RGB-only 不够。

最新 reach-only 证据把这个判断再往前推了一步：

> 当前 RLBench reach 的 RGB-only 已经 `1/1`。第一版 RGB-D normal 失败但 null 成功，说明 depth 会错误干预控制；safe RGB-anchor 版本 normal/null/cross-sample 全部 `1/1` 且动作诊断完全相同，说明低 gate + clamp 成功保护了 RGB policy，但 depth 内容仍未因果进入 action。

因此短期假设更新为：

> reach-only 只能做 sanity/regression。下一步不是继续在 reach 上调残差，而是把 normal/null/cross-sample causal gate 移到更 3D-sensitive 的任务和显式 action-space grounding 上，例如 keypose-conditioned residual、projected heatmap 或 coarse-to-fine 3D action map。

## 0.3 2026-07-04 当前实施更新

我们现在不再把问题描述成“LIBERO 数据集不好，所以换数据集”。更准确的版本是：

> LIBERO clean 已经被 RGB-only 刷到接近天花板，所以不适合做 depth gain 主证明；但在 RLBench `open_drawer` 上，单纯换数据集也还不够。真正要解决的是让 depth 的空间预测进入 action formation。

已经完成的关键证据：

- `reach_target` RGB-only 可以闭环成功，说明 RLBench action adapter 不是完全坏的。
- `reach_target` safe RGB-D normal/null/cross-sample 全成功且动作几乎一致，说明 shallow safe fusion 会忽略 depth。
- `open_drawer` RGB-only 没有刷满，且 offline dense-depth keypose probe 通过，说明这是更合适的 3D-sensitive gate。
- `open_drawer` safe RGB-D normal/null/cross-sample 仍然全部失败且 normal 与 cross-sample 动作几乎一样，说明浅层 residual 不够。

当前代码已经加入并验证了第一版显式 spatial-action coupling：

```text
dense point depth tokens
  -> object/action query attention
  -> absolute keypose prediction
  -> bounded keypose-conditioned action residual
  -> final delta action
```

对应参数：

- `--depth_keypose_residual_weight`
- `--depth_keypose_residual_clip`

`open_drawer` gate 结果：

- 从 RGB-only anchor resume。
- 冻结 VLA LoRA、proprio projector 和 action-head base。
- 开启 keypose-conditioned residual，保留 absolute keypose aux loss。
- 训练稳定到 `5000` steps。
- Paired normal-vs-cross-sample diagnostic 失败：`paired_pred_l1=0.0`，`paired_pred_rmse=0.0`，`paired_pred_xyz_l2=0.0`。
- Strict diagnostic 也没有分离：normal RMSE `0.0137849`，null `0.0137148`，cross-sample `0.0137849`。

结论：

> keypose residual 训练是稳定的，但它仍然没有让 action prediction 因果依赖 depth 内容。问题不是单纯 LIBERO 或数据量，而是 residual/keypose output 对 final action 仍然太可选。

因此下一步不是继续调 LIBERO 或继续加数据，而是切到更强的空间动作表示：

- BridgeVLA-style projected heatmap。
- Act3D/PerAct-style 3D action map。
- PointACT-style multi-scale point-action attention。
- 或者 ManiSkill3 adapter，用高吞吐 RGB-D/point cloud 数据先训练 3D action module。

## 0.4 2026-07-04 新方法搜索补充

新增搜索到的 2025/2026 方向进一步支持当前判断：

| 方法 | 关键信号 | 对本项目的影响 |
|---|---|---|
| Any3D-VLA | 显式把输入 lift 成 point cloud，比隐式空间 prior 更适合细粒度空间关系 | 继续用 dense point tokens，而不是回到 RGB-only 或浅层 depth embedding |
| PointACT | 让 action tokens 直接 attend 多尺度 point cloud，强调 local geometry + global scene structure | 我们的 keypose residual 是轻量版 action-token/point interaction；若失败，应升级多尺度 point-action attention |
| OG-VLA | 把 quasi-static manipulation 分解为 end-effector keyframes | 支持 absolute keypose / keyframe action target，不只训 short-horizon delta |
| 3DS-VLA | 用 3D spatial constraints 对齐 affordance-relevant objects 和 robot actions | 支持 task/object-conditioned 3D supervision，而不是无条件 depth summary |

参考链接：

- Any3D-VLA: https://arxiv.org/html/2602.00807v2
- PointACT: https://roboticsconference.org/program/papers/73/
- OG-VLA: https://og-vla.github.io/
- 3DS-VLA: https://proceedings.mlr.press/v305/li25g.html

这轮搜索后的工程结论：

> 下一步不应该再问“还要不要 LIBERO”，答案已经是不把它当主 benchmark。真正的问题是：我们能不能让 depth/point cloud 和 action token/keypose 发生强耦合。当前 keypose-conditioned residual 已经作为最低成本验证失败；升级路线应该是 multi-scale point-action attention 或 projected/3D action map。

## 0.5 2026-07-04 最新 no-go 与路线再收紧

最新 point-action gate 也失败了，所以路线需要再收紧一次。

已完成的新增实现：

```text
dense point depth tokens
  -> action/language-conditioned point scoring
  -> soft selected 3D point
  -> point_keypose_xyz auxiliary prediction
  -> bounded point-selected translation residual
  -> final delta action
```

训练设置：

- task：`open_drawer`
- data：`/root/RLBench/rgbd_hdf5_open_drawer_3demos_64`
- steps：`5000`
- aux：`DEPTH_AUX_TARGET=point_keypose_xyz`
- point residual：`DEPTH_POINT_ACTION_WEIGHT=1.0`，`DEPTH_POINT_ACTION_CLIP=0.02`
- 从 RGB-only anchor resume，并冻结 RGB/LoRA/proprio/action-head base。

关键修复：

- RLBench intrinsics 的 `fx/fy` 可能为负；旧 backprojection 把负 focal length clamp 到 `1e-6`，会让 point cloud 爆到 `1e8` 量级。
- 修复后 point cloud 回到米级，真实 `open_drawer` 单步 smoke 的 point aux loss 约为 `0.0327`。

结果：

| diagnostic | value |
|---|---:|
| `paired_pred_l1` | `0.0` |
| `paired_pred_rmse` | `0.0` |
| `paired_pred_xyz_l2` | `0.0` |
| `paired_pred_rpy_l2` | `0.0` |
| `paired_pred_gripper_abs` | `0.0` |
| single-step `xyz_rmse` | `0.001285` |

判定：

> NO-GO。即使让 action/query 去 attend point cloud，并把 selected point 接成 bounded residual，final action 仍然完全不随 normal/cross-sample depth 改变。

这说明问题已经不是“有没有 point cloud 表示”，而是：

> 只要 RGB action expert 仍然可以作为主路径，任何 depth/keypose/heatmap/point output 都可能退化成可选 residual。下一步必须让 3D action/waypoint 成为主 action output，而不是 correction。

最新联网搜索后的数据集判断：

- **ManiSkill3**：官方仓库说明支持 GPU parallel visual data collection，4090 上可采集 RGBD + segmentation 到 `30,000+ FPS` 量级，还支持 pointcloud/voxel visual input。它适合在我们需要更多 RGB-D 数据或更快 action-map 预实验时接入。
- **RoboCasa365**：官方 repo 2026-02-18 发布 v1.0，包含 `365 tasks`、`2500+ kitchen scenes`、`2200+ hours` demonstration 和 benchmark 支持。它适合后期大规模 household 泛化，不适合第一轮最小 gate。
- **RoboTwin 2.0**：官方 repo 已列出 DP、ACT、DP3、RDT、PI0、OpenVLA-OFT 等 baseline 支持，并提供数据采集、leaderboard 和预采集数据入口。它适合后续强 domain randomization/双臂路线。
- **DROID / Open X-Embodiment**：规模很强，适合真实数据预训练，但 closed-loop causal gate、depth modality 一致性和 action space 对齐成本更高，不应该作为当前第一步。

最新方法判断：

- **Act3D** 的关键不是“用了 3D”，而是把 6-DoF keypose prediction 当作 3D detection/action map。
- **PointACT** 的关键不是“point cloud 作为额外输入”，而是 action tokens 直接、多尺度地和 point cloud 交互。
- **DP3 / 3D Diffusion Policy** 的关键不是大模型，而是 sparse point cloud + compact 3D encoder + 直接生成 action trajectory。

因此下一步工程路线应改成：

```text
方案 A：primary 3D waypoint head
  - action/query score dense 3D candidate points
  - selected xyz 直接作为下一 waypoint / translation target
  - xyz loss 直接打在 selected 3D target 上
  - final action 的 xyz 来自 selected target - current ee xyz
  - RGB base action 只保留 rpy/gripper 或作为 fallback ablation

方案 B：Act3D-style coarse 3D action map
  - workspace coarse grid / sampled free-space points
  - query 根据 RGB/language/proprio/depth 打分 3D candidates
  - coarse-to-fine refine selected 3D action
  - normal/cross-sample 必须改变 selected 3D cell

方案 C：DP3-style 3D diffusion/action chunk
  - point cloud encoder 输出 compact 3D token
  - diffusion/action decoder 直接生成 delta action chunk
  - normal/null/cross-sample gate 放在生成动作差异上
```

当时的最小建议：

## 0.6 2026-07-04 最后 learned positive diagnostic 后的更新

ManiSkill3 PickCube 最后一轮给出了一个更具体的路线判断：

```text
geometry controller:
  normal 8/10
  null 0/10
  cross_demo 1/10

object-feature MLP on official demos:
  offline normal-vs-cross gate strong
  closed-loop 0/10

teacher-distilled object-feature MLP without phase:
  normal reward high but success 0/10
  debug: grasps cube but fails move-goal transition

phase-conditioned teacher-distilled MLP:
  normal 17/30
  null 0/30
  cross_demo 0/30

phase/geometry disentanglement:
  normal geometry + normal phase 6/10
  null geometry + normal phase 0/10
  cross geometry + normal phase 0/10
  normal geometry + null phase 0/10
```

这个结果改变了问题的表述：

> 不是“depth 有没有用”，也不只是“object-centric geometry 能不能进 action”。真实 pointcloud geometry 已经能驱动 learned action decoder 成功控制；失败的关键是 temporal state / phase inference。

因此下一步方法优先级应调整为：

1. **ACT-style temporal aggregation / action chunking**：让模型从一段 observation/action history 中学习 phase，而不是靠单步 MLP。
2. **DP3 / diffusion action policy**：用 pointcloud/object features 条件化 action trajectory，直接建模多步控制分布。
3. **Recurrent or latent-phase policy**：显式学习 approach / descend / close / lift / move-goal 这种隐状态。
4. **DAgger / teacher distillation**：继续用 geometry controller 生成更多 normal pointcloud 成功轨迹，但把 phase 从 teacher label 逐步蒸馏成模型内部状态。
5. **Act3D/PerAct action map**：如果转到更复杂任务，再用 3D action map 学 waypoint；但 PickCube 当前最短板是时序，而不是 3D candidate availability。

答辩口径：

> 最后一轮不是 OpenVLA RGB-D 正结果，因为 phase 仍由手写状态机提供；但它是 learned positive diagnostic。normal pointcloud `17/30`，null/cross `0/30` 证明了：在合适的 object-centric representation 和 temporal state 下，RGB-D/pointcloud 可以超过 no-depth/corrupt-depth controls。Phase/geometry 解耦进一步说明成功不是 phase alone，真实 object geometry 也必不可少。下一步应把手写 phase 换成 ACT/DP3/recurrent policy 自己学到的 temporal state。

> 先不要接 RoboCasa/RoboTwin，也不要继续 residual sweep。下一步在现有 RLBench `open_drawer` 上做 **primary 3D waypoint head**，因为它复用当前 HDF5、point cloud、keypose label、diagnostic，改动最小，而且能直接测试“selected 3D point 是否改变 final translation”。

这个建议已经执行，结果见下一节：primary waypoint-action 仍然 no-go。

## 0.6 2026-07-04 最后尝试：primary waypoint-action 仍然 no-go

上述 primary waypoint head 已经完成一次最后尝试。实现上不再让 selected 3D point 只是 residual，而是通过 `DEPTH_WAYPOINT_ACTION_WEIGHT=1.0` 直接覆盖 first-step xyz action，并用 `DEPTH_WAYPOINT_ACTION_CLIP=0.02` 控制步长。

结果：

| check | value |
|---|---:|
| paired normal-vs-cross `paired_pred_l1` | `5.90e-05` |
| paired normal-vs-cross `paired_pred_rmse` | `1.15e-04` |
| paired normal-vs-cross `paired_pred_xyz_l2` | `3.05e-04` |
| strict normal `xyz_rmse` | `0.003178` |
| strict null `xyz_rmse` | `0.003210` |
| strict cross-sample `xyz_rmse` | `0.003226` |
| rollout normal | `0/1`, length `11`, `InvalidActionError` |
| rollout null | `0/1`, length `10`, `InvalidActionError` |
| rollout cross-sample | `0/1`, length `11`, `InvalidActionError` |

判定：

> 这是 OpenVLA/RLBench waypoint recipe 的 no-go：selected 3D point 直接进入 final xyz action 后，normal/cross-sample action 不再完全相同，但差异太小，并且 closed-loop 全部失败。因此这段不能作为 RGB-D 超过 RGB 的主证据；它不证明 depth 没用，只说明当前 OpenVLA action path 没有稳定因果使用 depth。

最终汇报建议：

- 主贡献放在完整 RGB-D pipeline、严格 causal ablation、LIBERO 饱和判断、RLBench `open_drawer` 3D-sensitive gate 和关键 bug fix。
- 主要失败原因表述为：depth 有离线空间信号，但当前行为克隆和 OpenVLA-OFT action path 没有把该信号变成必要动作信息。
- 未来工作不要继续快速 patch residual/waypoint；应该换成更彻底的 3D action-space redesign，例如 Act3D/PerAct-style coarse-to-fine action map、DP3-style 3D action decoder，或在 ManiSkill3 上用更高吞吐数据先训练 3D action module。

## 0.7 2026-07-04 官方来源复核

最后一轮联网复核后，外部依据仍然支持当前结论：

| 来源 | 复核到的信息 | 对项目的含义 |
|---|---|---|
| RLBench official site / paper | RLBench 是 100 个手工任务的大规模 vision-guided manipulation benchmark，包含 RGB、depth、segmentation、proprioception 和 motion-planner demos | 保留为新 action decoder 的 regression/diagnostic benchmark，而不是回到 clean LIBERO |
| ManiSkill3 docs | 官方支持 `rgb+depth` / `pointcloud` observation、trajectory replay conversion、delta/non-delta EE control，以及 GPU parallel visual data throughput | 已成为本项目最终正结果主线，支撑从 `30` 到 `100` 条 teacher 轨迹的 pointcloud scaling |
| CALVIN dataset README | 官方说明 inference 支持 absolute actions `((x,y,z), euler, gripper)` 和 relative 7D actions | action-space 对照不能只做 delta；absolute/keypose action 仍值得作为下一轮设计 |
| PerAct project/page | RGB-D voxel observation + language，输出 discretized next-best voxel action | depth 最有效时通常和 3D action space 绑定，而不是作为可选 feature |
| Act3D project/paper | 把 6-DoF keypose prediction cast 成 3D detection，并 coarse-to-fine 计算高分辨率 3D action map | 下一轮应做真正 3D action map，而不是再叠 residual |
| DP3 project/paper | sparse point cloud + compact 3D encoder + diffusion action sequence | 如果不继续改 VLA head，DP3-style 3D action decoder 是更自然路线 |
| BridgeVLA paper/repo | point cloud 投影成多视角 2D images，并预测 2D heatmaps 估计 translational action | 我们的 heatmap probe 方向合理，但要完整 input-output alignment，而不是只把 heatmap 接 residual |
| RoboCasa365 official/repo | 365 tasks、2500+ kitchen scenes、2200+ hours demo/benchmark support | 适合第二阶段泛化验证，不适合明天前临时接入 |
| RoboTwin 2.0 repo | 支持 DP、ACT、DP3、RDT、PI0、OpenVLA-OFT 等 baseline | 适合后续双臂/强 domain randomization，但当前接入成本高于 RLBench/ManiSkill |

复核后的最终判断：

> 数据集方面，RLBench/ManiSkill3 仍是最合理下一步；方法方面，必须做完整 3D action-space alignment。当前项目已经证明，小 residual、heatmap residual、point residual、primary waypoint patch 都不足以让 OpenVLA-OFT 稳定使用 depth。

## 0.8 2026-07-04 3D action-map feasibility probe

为了进一步判断 Act3D/PerAct-style action map 是否值得继续，实现了一个无需训练 7B 的候选点覆盖率 probe：

```text
experiments/robot/rlbench/probe_3d_action_map_feasibility.py
```

它比较三种 candidate 到 `rlbench_keypose_action[:3]` 的最近距离：

1. normal point cloud candidates。
2. cross-sample point cloud candidates。
3. current EE position fallback。

结果：

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

- normal point candidates 明显优于 cross-sample，说明真实 depth/point cloud 对 3D candidate coverage 有信号。
- 短步 `keypose/next_pose` label 太接近当前末端执行器，EE fallback 更好，本质上仍是短步 proprio target。
- future/final/farthest-future target 会显著削弱 EE shortcut，使 normal point candidates 同时优于 cross-sample 和 EE fallback。
- 这解释了为什么行为克隆下模型容易绕开 depth：如果只预测局部 delta，proprio/RGB shortcut 已经足够。

下一轮修正：

> 不只是换成 3D action map head，还要换 target。目标应是 object/contact-conditioned waypoint、task-level keypose、handle/contact point、或成功条件相关的 3D affordance，而不是当前 EE 附近的 next-pose label。

## 0.9 2026-07-04 最后一轮工程收束：long-horizon target 已接入

基于 3D action-map feasibility probe，训练入口已补上三种 long-horizon 3D auxiliary target：

- `future_pose_xyz`
- `final_pose_xyz`
- `farthest_future_pose_xyz`

其中 `farthest_future_pose_xyz` 最适合作为下一轮最小尝试，因为它在 `open_drawer` 和 stable6 probe 中都明显削弱了当前 EE fallback shortcut。它不是新的正结果，只是把下一轮实验的 action target 从“短步局部 delta”推向“更需要场景几何的 waypoint”。

验证状态：

```text
真实 open_drawer HDF5 上三种 target 都能生成 finite `(3,)` 标签。
RLBench stage runner 已暴露 `DEPTH_AUX_FUTURE_HORIZON`。
训练 dry-run 已确认 `--aux_future_horizon 10` 正确传入 `finetune_depthvla.py`。
`MAX_STEPS=1` 真实训练 smoke 通过，`farthest_future_pose_xyz` 的 prediction/label shape 均为 `(1, 3)`。
```

推荐下一轮第一个命令：

```bash
DEPTH_AUX_TARGET=farthest_future_pose_xyz \
DEPTH_AUX_OUTPUT_DIM=3 \
DEPTH_AUX_FUTURE_HORIZON=10 \
DEPTH_WAYPOINT_ACTION_WEIGHT=1.0 \
DEPTH_WAYPOINT_ACTION_CLIP=0.02 \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgbd
```

## 0.10 2026-07-04 500-step long-horizon target 结果

已经补跑 `farthest_future_pose_xyz` 的 500-step 小规模门槛实验：

| check | value |
|---|---:|
| paired normal-vs-cross `paired_pred_l1` | `2.65e-05` |
| paired normal-vs-cross `paired_pred_rmse` | `6.26e-05` |
| paired normal-vs-cross `paired_pred_xyz_l2` | `1.66e-04` |
| strict normal `xyz_rmse` | `0.003190` |
| strict null `xyz_rmse` | `0.003210` |
| strict cross-sample `xyz_rmse` | `0.003189` |

判定：

> NO-GO。把 auxiliary target 从短步 `point_keypose_xyz` 换成 `farthest_future_pose_xyz` 并没有让 final action 更依赖真实 depth；paired action delta 反而小于上一轮 primary waypoint-action。这个结果进一步说明，问题不只是 target horizon，而是当前 action path 仍然把 3D signal 压成弱扰动。

外部来源复核后，下一步不应继续 scale 这个 waypoint recipe：

- ManiSkill3 官方文档强调 GPU-parallel RGBD/segmentation visual data collection，可作为下一轮高吞吐 RGB-D 数据来源。
- RoboTwin 2.0 已有 OpenVLA-OFT 使用文档，适合后续 benchmark 扩展，但当前接入成本高于 RLBench/ManiSkill3 pilot。
- Act3D 把 6-DoF keypose prediction cast 成 3D detection/action-map，这比当前 selected-point waypoint patch 更彻底。
- DP3 使用 sparse point cloud 的 compact 3D representation 直接生成 action sequence，说明“3D 表示 + action decoder”应一起换。

参考：

- ManiSkill3 docs: https://maniskill.readthedocs.io/
- RoboTwin OpenVLA-OFT docs: https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html
- Act3D: https://act3d.github.io/
- 3D Diffusion Policy: https://3d-diffusion-policy.github.io/

## 0.11 2026-07-04 5000-step long-horizon target 结果

提交前又补跑了 `farthest_future_pose_xyz` 的 5000-step 版本，目的是确认 500-step no-go 不是训练太短造成的。

结果：

| check | value |
|---|---:|
| paired normal-vs-cross `paired_pred_l1` | `1.58e-05` |
| paired normal-vs-cross `paired_pred_rmse` | `3.79e-05` |
| paired normal-vs-cross `paired_pred_xyz_l2` | `1.00e-04` |
| strict normal `xyz_rmse` | `0.003167` |
| strict null `xyz_rmse` | `0.003210` |
| strict cross-sample `xyz_rmse` | `0.003162` |

判定：

> NO-GO。训练到 `5000` steps 后，normal-vs-cross 的 action delta 进一步变小，strict diagnostic 中 cross-sample 还略优于 normal。这说明问题不是简单训练步数不足，而是当前 waypoint recipe 没有把 depth 变成 action formation 的必要变量。

最终路线收紧为：

```text
不继续扩大当前 waypoint/residual recipe。
下一轮若继续做正结果，应改成真正的 3D action map、object/contact-conditioned waypoint，或 DP3-style point-cloud action decoder。
```

## 1. 为什么 RLBench 仍然有诊断价值

RLBench 官方定位是大规模视觉机器人操作 benchmark，覆盖 reinforcement learning、imitation learning、multi-task learning、geometric computer vision 和 few-shot learning，并包含约 100 个手工设计任务。

更关键的是，PerAct、RVT/RVT-2 等工作都把 RLBench 当作主要 3D manipulation benchmark：

- PerAct 使用 RGB-D voxel observation，预测 6-DoF keyframe action，在 18 个 RLBench 任务、249 个 variation 上做多任务 behavior cloning。
- RVT 使用 multi-view transformer 和 workspace re-rendering，在同样的 18-task RLBench 设置上验证 3D/multi-view 表示，并报告比 PerAct 更高的相对成功率和更快训练/推理。
- 这些方法都不是简单把 depth 拼到 RGB 后面，而是把 observation representation 和 action representation 显式绑定到 3D 空间。

因此 RLBench 对我们最有价值的地方不是“任务多”或“继续扩大旧 recipe”，而是：

1. 它有 RGB-D、多相机、相机参数、gripper pose、语言描述和 closed-loop success。
2. 它的经典任务需要插入、堆叠、抽屉、旋钮、容器等 3D 几何。
3. 已有强 3D 方法给出清晰设计参考，可以直接指导 DepthVLA 的 action-space 和 fusion 改造。
4. 本仓库已经打通 RLBench HDF5、offline probe 和 rollout，可作为新 action decoder 的 regression gate。

## 2. 数据规模路线

不要一上来跑 MT100。建议分三阶段：

| 阶段 | 任务数 | demos/task | 目标 |
|---|---:|---:|---|
| Pilot | 6 | 10 | 跑通转换、probe、RGB/RGB-D 训练、normal/null/shuffle |
| Stage 1 | 18 | 10 | 复现 PerAct/RVT 常用多任务设置，观察 RGB-D 是否有方向性收益 |
| Stage 2 | 18 | 100 | 扩大训练量，验证收益是否稳定 |
| Stage 3 | 55/100 | 10-50 | 多任务泛化，不作为第一轮成功标准 |

第一批稳定 pilot 6 个任务，已用 1-demo live collection smoke 验证：

```text
slide_block_to_target
turn_tap
close_jar
open_drawer
reach_target
pick_up_cup
```

选择这些任务的原因：

- `slide_block_to_target`、`turn_tap`、`close_jar`、`open_drawer` 都包含目标位置、接触或 articulation 几何。
- `reach_target` 是低风险 sanity task，用来确认 RGB/RGB-D 管线和动作空间没有系统性错误。
- `pick_up_cup` 引入 grasp/contact，比纯 reach 更接近 manipulation。

后续扩展候选：

```text
insert_onto_square_peg
stack_blocks
put_item_in_drawer
place_wine_at_rack_location
```

这些任务 3D 价值更强，但当前 CoppeliaSim/RLBench fork 的 live-demo smoke 中存在 asset handle 或 planning 问题，不能作为默认 pilot。
- `turn_tap` 需要接触点和旋转几何。
- `slide_block_to_target` 能暴露目标位置和相对平面几何。

## 3. Action Space 结论

只输出短 horizon delta action，depth 很容易被 RGB/proprio shortcut 绕过。PerAct 的核心启发是：把动作预测改成 3D 空间中的 next-best keyframe/action detection；CALVIN 的价值是提供 absolute 和 relative cartesian action 的对照接口。

推荐下一轮不是“彻底放弃 delta”，而是：

```text
执行头：delta action chunk
空间辅助头：absolute keypose / 3D heatmap / action map
训练目标：normal depth 必须能预测下一关键位姿，null/shuffle 不能做到
```

这样做的好处：

1. 保留 OpenVLA-OFT 当前 delta chunk 的闭环稳定性。
2. 用 absolute keypose 把 depth 表示和 metric 3D action target 对齐。
3. 用 normal/null/shuffle probe 先筛掉“看似能训练但不因果使用 depth”的方案。

当前仓库已经落实第一版：

- RLBench converter 保存 `rlbench_keypose_action`，8 维：下一绝对 gripper pose + gripper open。
- `finetune_depthvla.py` 支持 `--aux_target absolute_keypose --aux_output_dim 8`。
- `probe_dense_depth_keypose.py` 用 normal/null/shuffle 测试 dense depth tokens 是否真的能预测 absolute keypose。

## 4. Depth Fusion 结论

旧的 grid-pooled summary 问题是过早压缩空间结构，导致 normal 与 shuffle depth 差异不足。下一轮应使用更接近 3D policy 文献的表示：

1. **Dense 3D point tokens**
   - 从 depth + K/T 反投影到 base/world frame。
   - 采样 1024-4096 个点。
   - feature 包含 xyz、EE-relative xyz、z_cam、uv、view id、valid mask。

2. **Object/action query cross-attention**
   - 用语言、proprio 和 action hidden state 形成 query。
   - query attend dense 3D tokens，而不是全局平均池化。

3. **Absolute keypose / heatmap auxiliary**
   - 训练时强制 depth branch 解决空间定位问题。
   - rollout 时仍以 delta chunk 执行，避免 absolute control 不稳定。

当前仓库已新增：

```text
prismatic/models/dense_point_depth_encoder.py
```

并在训练脚本中支持：

```text
--depth_encoder_type dense_point
--depth_num_points_per_view 1024
--depth_integration_mode depth_object_query
--aux_target absolute_keypose
```

## 4.1 当前真实 RLBench Pilot 证据

已生成并转换两个真实 pilot：

| 数据 | demos/task | transitions | image_size | HDF5 |
|---|---:|---:|---:|---|
| stable6 smoke | `1` | `632` | `64x64` | `/root/RLBench/rgbd_hdf5_stable6_1demo_64` |
| stable6 scaled pilot | `3` | `2009` | `64x64` | `/root/RLBench/rgbd_hdf5_stable6_3demos_64` |

任务集合：

```text
slide_block_to_target, turn_tap, close_jar, open_drawer, reach_target, pick_up_cup
```

Dense depth -> absolute keypose offline probe 已通过。最新 `3 demos/task` 结果使用更严格的跨样本 shuffle 消融：目标来自样本 A，但 depth/camera/mask 来自样本 B。

| depth 输入 | keypose xyz RMSE |
|---|---:|
| normal | `0.0803` |
| null | `0.6097` |
| shuffle_samples | `0.2035` |

对照 1-demo smoke 结果：normal `0.0204`，null `0.2274`，shuffle `0.0652`。

这个结果说明：

> 在 RLBench RGB-D 数据上，normal metric depth 对 absolute keypose 预测有明确因果信息；因此下一步值得进入 matched RGB-only / RGB-D VLA action training。

训练入口状态：

- RGB-only `MAX_STEPS=1` smoke：通过。
- RGB-D dense point + object-query + absolute keypose `MAX_STEPS=1` smoke：通过。
- RGB-only / RGB-D RLBench rollout smoke：通过。

注意：

> 这只是 RLBench 阶段性 gate，不是最终 claim。当前只证明数据、offline gate、训练入口和 rollout 入口成立；还需要 matched closed-loop rollout，才能判断这条 RLBench/OpenVLA recipe 是否超过 RGB baseline。

## 4.2 当前 matched rollout 结果

已完成第一组 matched 小训练：

| 模型 | 数据 | steps | checkpoint |
|---|---|---:|---|
| RGB-only | stable6 `3 demos/task` | `2000` | `/root/runs_rlbench_stable6_3demos/...--rlbench-rgb-only` |
| RGB-D dense point | stable6 `3 demos/task` | `2000` | `/root/runs_rlbench_stable6_3demos/...--rlbench-rgbd-dense-keypose` |

第一版小规模 closed-loop eval 设置：

```text
6 tasks
1 episode/task
25 max steps/episode
image_size: 64x64
```

结果：

| policy / depth mode | success |
|---|---:|
| RGB-only | `0/6` |
| RGB-D normal | `0/6` |
| RGB-D null | `0/6` |
| RGB-D shuffle | `0/6` |

Causal gate：`NO-GO`。

后续诊断发现：这组 `25 max steps` 不能作为有效 rollout 结论，因为真实 demo 的 next-absolute-pose replay 在相同 planning action mode 下需要更长 horizon 才能成功：

| 任务 | replay 成功步数 |
|---|---:|
| `slide_block_to_target` | `80` |
| `turn_tap` | `136` |
| `close_jar` | `120` |
| `open_drawer` | `89` |
| `reach_target` | `30` |
| `pick_up_cup` | `91` |

也就是说，RLBench eval 的默认 horizon 应该至少设到 `150`，否则会把“episode 太短”误判成“policy 完全失败”。

`150 max steps` matched eval 当前结果：

| policy / depth mode | success | 主要现象 |
|---|---:|---|
| RGB-only | `0/6` | 所有任务在 `19-58` 步左右触发 `InvalidActionError` |
| RGB-D normal | `0/6` | 所有任务在 `28-96` 步左右触发 `InvalidActionError` |

Reach-only trace：

| policy / depth mode | success | length | delta xyz mean | gripper mean |
|---|---:|---:|---:|---:|
| RGB-only | `0/1` | `59` | `0.01435` | `1.0000` |
| RGB-D normal | `0/1` | `70` | `0.01109` | `0.8566` |
| RGB-D null | `0/1` | `72` | `0.01135` | `0.8956` |

Offline policy-vs-demo action diagnostic 进一步说明了问题：

| policy / depth mode | samples | xyz RMSE | xyz direction cosine |
|---|---:|---:|---:|
| RGB-only | `12` | `0.00564` | `0.9259` |
| RGB-D normal | `12` | `0.00514` | `0.6051` |
| RGB-D null | `12` | `0.00515` | `0.6150` |
| RGB-D shuffle | `12` | `0.00514` | `0.6051` |

当前更准确的解释：

> Offline keypose probe 已经证明 dense depth 对 absolute keypose 有信号；demo replay 也证明 RLBench action mode 基本可执行。但 policy-vs-demo 诊断显示 RGB-D normal/null/shuffle 的 action prediction 几乎重合，说明当前 VLA action head 还没有因果使用 depth 内容。当前瓶颈是 depth-action coupling 加 closed-loop execution stability，而不是继续刷 LIBERO 或盲目扩大同一个 recipe。

Reach-only 最新 gate：

| policy / depth mode | MAX_DELTA_XYZ | success | length |
|---|---:|---:|---:|
| RGB-only | `0.03` | `1/1` | `29` |
| RGB-only | `0.05` | `1/1` | `29` |
| RGB-only | `0.08` | `1/1` | `29` |
| RGB-D normal | `0.03` | `0/1` | `150` |
| RGB-D normal | `0.05` | `0/1` | `150` |
| RGB-D normal | `0.08` | `0/1` | `150` |
| RGB-D null | `0.05` | `1/1` | `31` |
| RGB-D shuffle | `0.05` | `0/1` | `150` |

这说明 RGB-only action adapter 已经能在 RLBench 最小闭环任务上成功；失败点转移到了 RGB-D fusion。更具体地说，normal depth 不是没有影响，而是以当前方式进入 action head 后会把策略拉偏；null depth 反而保留了可成功的动作模式。

Safe RGB-anchor 修复结果：

| policy / depth mode | MAX_DELTA_XYZ | success | length | 诊断 xyz RMSE | 诊断 direction cosine |
|---|---:|---:|---:|---:|---:|
| RGB-D safe normal | `0.05` | `1/1` | `29` | `0.001700` | `0.97756` |
| RGB-D safe null | `0.05` | `1/1` | `29` | `0.001700` | `0.97756` |
| RGB-D safe cross-sample | `0.05` | `1/1` | `29` | `0.001700` | `0.97756` |

这个结果的正确解读是：

> 低 gate、clamp、冻结 RGB anchor 的方案修掉了“depth 伤害已成功 RGB policy”的问题；但这段本身不能证明 depth 有用，也不能证明 depth 无用。normal/null/cross-sample 的 rollout 和 action diagnostic 完全重合，所以只能说明这条 safe residual 路线里 depth 内容仍然没有因果进入 action。

更新后的下一步优先级：

1. 把 `reach_target` 降级为 sanity/regression gate：以后只要求 RGB-D 不破坏 RGB policy，不再把 reach 成功当 depth claim。
2. 下一轮选择更 3D-sensitive 的任务：`open_drawer`、`turn_tap`、`slide_block_to_target`、`pick_up_cup`，或 ManiSkill 中带高度/视角/物体位姿扰动的任务。
3. 不继续简单扩大 safe residual；必须加入显式空间动作监督：keypose-conditioned residual、projected heatmap 或 3D action map。
4. 只有 normal depth 在这些任务上超过 null/cross-sample，才进入 `6 tasks x 10 demos` 或 `18-task` scaling。

`open_drawer` 已作为下一轮候选 gate 启动：

| 项目 | 结果 |
|---|---:|
| HDF5 子集 | `3 demos / 317 transitions` |
| normal keypose xyz RMSE | `0.0705` |
| null keypose xyz RMSE | `0.1420` |
| shuffle keypose xyz RMSE | `0.1466` |

这说明 `open_drawer` 的 depth 对 absolute keypose 有明显离线因果信号，值得先训练 matched RGB-only baseline，再做 safe RGB-D 对照。

RGB-only baseline 已完成：

| 设置 | 结果 |
|---|---:|
| train steps | `5000` |
| eval horizon | `200` |
| `MAX_DELTA_XYZ` | `0.05` |
| success | `0/1` |
| length | `200` |

这个失败不是 `InvalidActionError`，而是 timeout；因此 `open_drawer` 比 reach 更适合作为 depth causal gate。下一步是从这个 RGB-only checkpoint resume，训练 safe RGB-D normal/null/cross-sample 对照。

Safe RGB-D 对照已完成，结论是 `NO-GO`：

| policy / depth mode | success | length | delta xyz mean | 诊断 xyz RMSE | 诊断 cosine |
|---|---:|---:|---:|---:|---:|
| RGB-only | `0/1` | `200` | `0.00595` | - | - |
| RGB-D normal | `0/1` | `200` | `0.00592` | `0.001336` | `0.78733` |
| RGB-D null | `0/1` | `200` | `0.00597` | `0.001357` | `0.78683` |
| RGB-D cross-sample | `0/1` | `200` | `0.00608` | `0.001336` | `0.78733` |

这组结果很关键：

> `open_drawer` 证明了“换出 LIBERO/换出 reach”是必要的，因为 RGB-only 不饱和，且 depth 对 keypose 有离线信号。但 shallow safe residual 仍然不够，因为 normal 和 cross-sample action diagnostic 完全相同。下一步要让 spatial prediction 直接参与 action，而不是只作为辅助 loss。

## 5. Go / No-Go 标准

先过 offline gate，再跑大训练：

```text
normal keypose xyz RMSE 至少比 null 低 0.01m
normal keypose xyz RMSE 至少比 shuffle 低 0.01m
```

最终 rollout gate：

```text
RGB-D normal - RGB-only >= 0.05 success rate
RGB-D normal - RGB-D null >= 0.05 success rate
RGB-D normal - RGB-D shuffle >= 0.05 success rate
```

如果 RGB-D normal 只超过 RGB-only，但没有超过 null/shuffle，不能 claim depth 有因果贡献。

## 6. 候选数据集对比

| 数据集 | 用途 | 优点 | 风险 |
|---|---|---|---|
| ManiSkill3 | 下一轮正结果主线 | GPU 并行、RGBD/segmentation/point cloud 采集快，适合 action decoder pilot | 和当前 OpenVLA/LIBERO/RLBench 管线差异大，需要新 adapter |
| RLBench | regression / diagnostic benchmark | RGB-D、相机参数、closed-loop、3D tasks、已有 PerAct/RVT 参考，本仓库已打通 | CoppeliaSim/PyRep 环境安装麻烦，生成数据慢；当前 residual/waypoint recipe 已 no-go |
| CALVIN | action-space 对照 | 支持 absolute/relative cartesian action，长时序语言任务 | depth 边际收益未必最直接 |
| RoboTwin 2.0 | 合成数据/强扰动/双臂路线 | 数据采集、leaderboard、OpenVLA-OFT/DP3 等 baseline 支持 | 双臂和环境接入成本高，容易偏离当前单臂 OpenVLA-OFT 问题 |
| RoboCasa365 | 第二阶段泛化 | 365 tasks、2500+ scenes、2200+ hours demos、household kitchen tasks | 工程接入和资源成本高，不适合第一轮最小实验 |
| DROID / Open X-Embodiment | 真实数据预训练 | 数据规模和真实多样性强 | 不提供统一 closed-loop sim eval；depth/action/camera 格式不一定满足 normal/null/cross causal gate |

## 7. 参考资料

- RLBench: https://github.com/stepjam/RLBench
- RLBench project page: https://sites.google.com/view/rlbench
- PyRep: https://github.com/stepjam/PyRep
- PerAct: https://peract.github.io/
- RVT: https://robotic-view-transformer.github.io/
- RVT code: https://github.com/NVlabs/RVT
- ManiSkill docs: https://maniskill.readthedocs.io/en/latest/user_guide/index.html
- CALVIN: https://github.com/mees/calvin
- CALVIN dataset action interface: https://github.com/mees/calvin/blob/main/dataset/README.md
- RoboCasa: https://robocasa.ai/
- RoboCasa repo: https://github.com/robocasa/robocasa
- RoboCasa365 paper: https://openreview.net/forum?id=tQJYKwc3n4
- RoboTwin 2.0: https://github.com/robotwin-Platform/robotwin
- DROID: https://droid-dataset.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- Act3D: https://act3d.github.io/
- PointACT: https://roboticsconference.org/program/papers/73/
- 3D Diffusion Policy: https://3d-diffusion-policy.github.io/
