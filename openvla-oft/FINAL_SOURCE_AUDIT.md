# DepthVLA-OFT 官方来源与实验决策核对

更新时间：2026-07-05 UTC

## 1. 结论

联网复核后的结论和本地实验一致：

> depth/pointcloud 不是自动加到 VLA 上就会有用；它必须进入更 3D-sensitive 的 benchmark，并和 action space 对齐。当前最强本地正证据是 ManiSkill3 PickCube raw cropped pointcloud policy：normal `20/60`，eval-time null/cross `1/60`，matched sampled-RGB-only/null train baselines `1/60` / `3/60`。

## 2. 数据集选择依据

| 来源 | 官方信息 | 对本项目的决策 |
|---|---|---|
| ManiSkill3 observation docs | 支持 `rgb+depth`、`rgb+depth+segmentation`、`pointcloud` 等 observation mode | 适合把 RGB-D 直接转成 pointcloud，做 normal/null/cross causal gate |
| ManiSkill3 replay docs | 官方支持 replay demonstrations，并转换 observation/control modes，例如把 demo replay 成 pointcloud + EE control action | 支撑我们把 PickCube 官方 demo/teacher rollout 转成 pointcloud action dataset |
| ManiSkill3 action-space docs | controller 定义 action space；EE pose/pos controller 支持 delta 和 non-delta/absolute target-style control | 支撑下一步比较 delta action、absolute/keypose target、object/contact-conditioned waypoint |
| ManiSkill3 performance docs | 官方 benchmark 包含 RGB+Depth 并强调 GPU parallel rendering / visual data throughput | 比 RLBench 更适合快速扩大 RGB-D/pointcloud 数据量 |
| RLBench official repo | 大规模 vision-guided manipulation benchmark，支持 imitation/multi-task/geometric vision 等研究；task sets 可到 15/30/55/100 | 适合做 3D-sensitive regression/gate，但当前 OpenVLA residual/waypoint recipe 已 no-go |
| CALVIN official repo | 支持 RGB-D sensors，并明确列出 absolute cartesian pose、relative cartesian displacement、joint action | 作为 action-space 对照来源，提醒我们不要只测试 relative delta action |

## 3. Action / Fusion 方法依据

| 方法 | 官方/论文信息 | 对本项目的启发 |
|---|---|---|
| DP3 / 3D Diffusion Policy | sparse point cloud 经过 compact 3D encoder，再条件化 diffusion action generator | 下一步不要只做浅层 action chunk；要让 action sequence 直接条件化在 3D 表示上 |
| PerAct | RGB-D voxel observation 和 discretized 6-DoF next-best voxel action 共享 3D 空间 | depth 的收益来自 observation-action 同空间，而不是可选 feature |
| Act3D | 把 6-DoF keypose prediction 作为 3D detection，coarse-to-fine 生成高分辨率 3D action map | 如果回到 RLBench，应做真正 3D action map，而不是 waypoint residual patch |
| BridgeVLA | 将 point cloud 投影成多视角 2D image，并预测 2D heatmap 做 translational action | 我们的 heatmap 方向是对的，但必须做 input-output alignment，不能只把 heatmap 当 residual feature |
| PointVLA | 冻结 vanilla action expert，并用轻量模块注入 point cloud，降低对 2D pretrained policy 的扰动 | 解释了为什么要保护 RGB anchor；但本地结果说明只保护还不够，depth 还要进入 primary action decoder |
| SpatialVLA | Ego3D position encoding 和 adaptive spatial grids，用 spatial action tokens 输出动作 | 支持 robot-centric spatial/action token 方向，而不是全局 pooled depth summary |

## 4. 本地证据闭环

| 要求 | 本地证据 |
|---|---|
| 换出饱和 LIBERO | clean LIBERO RGB-only 已接近天花板；文档中已降级为 sanity/history |
| 联网搜索更合适数据集 | `RGBD_DATASET_ACTIONSPACE_RESEARCH.md`、`NEXT_RGBD_BENCHMARK_PLAN.md`、本文件 |
| 扩大训练量 | ManiSkill3 raw cropped pointcloud teacher 数据从 `30` 成功轨迹扩大到 `100` 成功轨迹 / `8388` transitions |
| RGB-depth/pointcloud 超过 RGB/no-depth baseline | normal pointcloud `20/60` vs sampled-RGB-only `1/60` vs null/proprio `3/60` |
| normal 优于 corrupt depth | same policy eval-time normal `20/60` vs null `1/60` vs cross-demo `1/60` |
| 证据边界 | 当前正结果是 ManiSkill teacher-distilled PointNet/pointcloud policy，不是 OpenVLA 端到端 RGB-D claim |

## 5. 可引用链接

- ManiSkill3 observation docs: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html
- ManiSkill3 trajectory replay docs: https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html
- ManiSkill3 action-space docs: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html
- ManiSkill3 performance docs: https://maniskill.readthedocs.io/en/latest/user_guide/additional_resources/performance_benchmarking.html
- RLBench official repo: https://github.com/stepjam/RLBench
- CALVIN official repo: https://github.com/mees/calvin
- DP3 project: https://3d-diffusion-policy.github.io/
- PerAct project: https://peract.github.io/
- Act3D project: https://act3d.github.io/
- BridgeVLA paper/repo: https://arxiv.org/html/2506.07961v2 and https://github.com/BridgeVLA/BridgeVLA
- PointVLA project: https://pointvla.github.io/
- SpatialVLA project: https://spatialvla.github.io/
