# DepthVLA-OFT 最终提交总结

更新时间：2026-07-05 UTC

## 1. 最终结论

本项目最终证明了一个更有用的结论：

> 在更 3D-sensitive 的 ManiSkill3 PickCube 设置下，真实 depth/pointcloud 几何可以被 learned policy 因果使用，并且明显超过 no-depth / cross-demo controls。

最强 raw-pointcloud learned action 结果是：

| setting | success |
|---|---:|
| learned raw pointcloud policy, normal input | `20/60` |
| same policy, eval-time null input | `1/60` |
| same policy, eval-time cross-demo input | `1/60` |
| matched sampled-RGB-only train baseline | `1/60` |
| matched null/proprio train baseline | `3/60` |
| learned cube + fixed controller, normal/null/cross | `22/30` / `1/30` / `0/30` |

这说明 depth/pointcloud 不是“没用”：在相同 `100` 条 teacher 数据、相同模型容量、相同 eval seeds 下，normal pointcloud 明显超过训练时没有 3D 几何的 matched baselines。LIBERO/OpenVLA-OFT 轻量 adapter 路线没有形成正结果，但它的价值是作为对照：它解释了为什么 saturated benchmark 和 optional residual fusion 会掩盖或绕开 depth。

提交前最后一轮 ManiSkill3 teacher-distillation 得到了连续正向证据：先用 hand-written phase 的 object-feature learned policy 在 PickCube 上达到 normal pointcloud `17/30`，null `0/30`，cross-demo `0/30`；随后把 phase 换成 learned phase classifier 后，normal 进一步达到 `19/30`，null/cross 仍为 `0/30`。再进一步，把输入推向 raw cropped pointcloud 后，扩大到 `100` 条成功 teacher 轨迹、训练 h256/10k 单步 action decoder，在两组 30-episode eval seed 上合计 normal `20/60`，null `1/60`，cross-demo `1/60`。把同一个 raw pointcloud 学出的 cube predictor 接到固定 geometry controller，normal 还能达到 `22/30`，null `1/30`，cross-demo `0/30`。这些结果证明了真实 RGB-D/pointcloud 几何可以被模型学出，并且能明显超过 no-depth/cross controls。

## 2. 为什么不能继续用 LIBERO 作为主结论

clean LIBERO 已经被 RGB-only 刷到接近天花板，RGB-only clean trained tasks 可达到 `15/15`。在这种 benchmark 上，即使 depth 有边际价值，也很难从 aggregate success rate 中看出来。

因此本项目后半段把 LIBERO 降级为 sanity check，改用 RLBench `open_drawer` 做 3D-sensitive gate。

## 3. 最后一次 OpenVLA/RLBench closed-loop 诊断

最后一次进入 OpenVLA/RLBench rollout 的尝试不是早期 waypoint no-go，而是一个更强的几何瓶颈诊断：

- 数据：`/root/RLBench/rgbd_hdf5_open_drawer_3demos_64`
- 任务：`open_drawer`
- 设计：selected 3D point 直接覆盖 `8` 步 action chunk 的 xyz
- 关键参数：`DEPTH_WAYPOINT_ACTION_WEIGHT=1.0`，`DEPTH_WAYPOINT_ACTION_CLIP=0.02`，`DEPTH_WAYPOINT_ACTION_SCALE=20.0`，`DEPTH_WAYPOINT_ACTION_CHUNK_LEN=8`

先用 `visible_object_point_xyz` 训练的 chunk8 checkpoint 已证明 depth 进入 temporal action geometry：

| 检查项 | 结果 |
|---|---:|
| paired normal-vs-cross `paired_depth_point_xyz_l2` | `0.1951` |
| paired normal-vs-cross `paired_depth_waypoint_chunk_xyz_action_l2` | `1.7038` |
| paired normal-vs-cross `paired_pred_xyz_l2` | `0.00678m` |
| strict normal / null / cross `xyz_rmse` | `0.00414` / `0.00603` / `0.00541` |
| true chunk execution normal | `0/1`, length `200`, no error |
| true chunk execution null | `0/1`, length `4`, `InvalidActionError` |

trace 显示 normal EE-to-selected-depth-point 从 `0.2780m` 降到 `0.0144m`，动作方向和 depth point 方向 cosine 均值 `0.8103`。也就是说，OpenVLA 这里已经不是“depth 没进 action”。

随后做了两个最后诊断：

1. `latch_close_near_depth_point`：normal 在接近 selected point 后能触发夹爪闭合，最小 EE-to-depth-point `0.00464m`，但仍 `0/1`。
2. `first_close_pose_xyz`：用 demo 第一次 gripper close 的 EE xyz 作为 contact supervision。paired normal-vs-cross 仍有 `paired_pred_xyz_l2=0.00659m`，第一步 selected point 到了把手高度附近 `z≈1.039`；但闭环中 selected point drift 达到 `0.3408m`，后续漂到错误接触区域。宽阈值强制闭合后只 pull 1 步即 `InvalidActionError`。
3. `depth_point_latch_mode` eval-only 诊断：固定 learned first selected point 后，EE-to-point 可从 `0.473m` 降到 `0.00062m`，step `45` 关闭并执行 `35` 步 `+Y` pull，rollout 能跑满 `200` 步但仍 `0/1`。进一步用 oracle demo first-close point 作为固定目标也未成功。因此剩余问题不只是 point drift，还包括 handle/contact pose、gripper orientation 和 post-contact pull trajectory。
4. `eval_demo_replay.py` oracle 诊断：用同一个 RLBench action mode 直接回放存储 demo EE pose/gripper 序列，`open_drawer` 三条 demo 均成功，`3/3`，平均长度 `92.33`。这说明当前 eval stack/planner 能执行专家轨迹，不是环境本身坏了。
5. `post_close_demo_tail` oracle upper bound：严格 close threshold `0.03/0.05/0.08m` 下 demo tail 都没有触发，说明直线 point approach 不稳定，不能可靠进入接触前状态；但把触发阈值放宽到 `0.20m` 并从 demo first-close 前 `20` 帧接管后，rollout `1/1` 成功，长度 `61`。这不是 learned policy success，而是说明如果 depth/geometric policy 能稳定进入可接管邻域，再接 temporal/contact trajectory 是有机会完成任务的。
6. `learned first selected point + demo tail` 诊断：把 oracle first-close point 换成模型第一步 selected depth point 后，normal 在 `0.20m` gate 下 `1/1`，长度 `64`；null `0/1`，第 `2` 步 `InvalidActionError`。但 cross-sample 和 shuffle 也能成功，所以这条不能作为 strict normal-vs-corrupt success，只能说明非空几何点加 expert tail 有接管上界，真正 learned policy 仍需要更强 object/contact grounding。
7. `pre_first_close_pose_xyz` 500-step 诊断：把 target 从 first-close 接触点前移到 demo-tail 成功接管的 pre-contact 入口后，normal 比 null 更好，但 cross-sample 仍然追上甚至超过 normal。strict `xyz_rmse` 为 normal `0.00382`、null `0.00491`、cross `0.00357`；`xyz_direction_cosine` 为 normal `0.588`、null `0.251`、cross `0.753`。因此这不是 OpenVLA learned task success，而是说明 pre-contact target 改善了 normal-vs-null，却仍未解决 object/contact-specific grounding。

判定：OpenVLA/RLBench 仍是 NO-GO learned task success，但失败边界已经收紧到“稳定进入接触前状态 + 可执行 gripper pose/orientation + post-contact temporal trajectory”，而不是 depth-action coupling 缺失，也不是 RLBench action stack 无法执行专家轨迹。

## 4. 项目贡献

1. 搭建了 RGB-D OpenVLA-OFT/RLBench 管线，包括数据转换、训练、诊断和 rollout。
2. 建立了 matched RGB-only baseline，并确认 clean LIBERO 不适合作为 depth gain 主证据。
3. 加入 normal/null/shuffle/cross-sample depth causal ablation。
4. 验证 depth 离线空间信号存在，例如 projected keypose heatmap probe：
   - `open_drawer`: normal `2.85px`，cross `8.86px`，null `43.30px`
   - stable6: normal `2.94px`，cross `12.53px`，null `57.63px`
5. 修复了 RLBench negative focal-length backprojection bug，避免 point cloud 爆到异常尺度。
6. 系统排除了多种轻量融合路线：safe residual、keypose residual、projected UV、projected heatmap residual、point-action residual、primary waypoint-action。
7. 新增 3D action-map feasibility probe，发现短步 keypose/next-pose label 有明显 EE shortcut；但 future/final/farthest-future target 能让 normal point candidates 同时优于 cross-sample 和 EE fallback，说明下一轮应改 target，而不是只改 head。
8. 新增 demo replay 与 demo-tail oracle 诊断，确认 RLBench 专家轨迹可执行，并给出 OpenVLA 后续路线的上界：depth/geometric policy 需要学到接触前状态和后续时序，而不是只输出单个 xyz 点。
9. 新增 `pre_first_close_pose_xyz` target：监督 close 前 `aux_future_horizon` 帧的 EE xyz，默认可设为 `20`，直接对齐 demo-tail 成功接管入口。它已通过 synthetic dataset smoke、真实 `open_drawer` label sanity、`MAX_STEPS=1` 真实训练 smoke、小样本 paired diagnostic 和 500-step gate。500-step 结果继续支持“depth 进入 action geometry”，但没有通过 strict normal-vs-cross 成功门槛。

## 5. OpenVLA/RLBench 路线瓶颈

针对 OpenVLA-OFT/RLBench 轻量路线，早期瓶颈是 depth-action coupling 不够强；最后的 chunk8/trace 诊断已经把这点推进了一步：depth 可以进入 temporal action geometry，但还不能稳定完成 contact-level manipulation。

在行为克隆下，如果 RGB、proprioception 和语言先验已经能降低 action loss，模型没有被强制使用 depth。即使加入 auxiliary spatial target，final action 仍可能绕开 depth，或者只产生很小、不稳定的动作扰动。

最后的 waypoint/action-chunk 实验说明：单纯把 selected 3D point 接到 xyz action 还不够。即使 eval-only latch 消除了 selected point drift，当前系统仍缺少可执行的 handle/contact pose、gripper orientation 和抓取-拉开的 post-contact 时序。demo replay `3/3` 和 `0.20m` demo-tail upper bound `1/1` 进一步说明任务不是不可执行；learned first selected point + demo tail 的 normal/null 差异说明完全移除几何会破坏这个接管路径，但 cross/shuffle 也成功，说明它还不是 strict depth causal success。真正缺的是 learned policy 稳定进入 object-bound 接触前状态并生成后续 temporal/contact action。这个 no-go 只能限制当前 recipe，不能推出 “OpenVLA RGB-D 没用”。

后续 3D action-map feasibility probe 进一步说明：如果 target 仍然是接近当前 EE 的 next pose/keypose，即使实现 3D action map，模型也可能继续靠 proprioception 学到短步动作；而 future/final/farthest-future target 明显更能体现 point-cloud/depth 价值。下一轮需要 object/contact-conditioned waypoint 或真正 task-level 3D action target。

## 6. 最后一轮收束

在提交前，代码已补上 long-horizon 3D auxiliary target：

- `future_pose_xyz`
- `final_pose_xyz`
- `farthest_future_pose_xyz`
- `pre_first_close_pose_xyz`

这一步没有形成新的正结果，因此不应写成 improvement。它的价值是把失败定位进一步收紧：上一轮 `point_keypose_xyz` 太容易被 EE/proprio shortcut 解释，所以提交前又测试了 `farthest_future_pose_xyz`，继续用 normal/null/cross-sample gate 判断是否真的使用 depth。

最后又补了一个更贴合 demo-tail upper bound 的 `pre_first_close_pose_xyz` target：在 `open_drawer` demo0 中，first close index 是 `74`，`aux_future_horizon=20` 对应 pre-close index `54`，xyz 为 `[0.2118, 0.0601, 1.0512]`，正好是前面 demo-tail 成功接管使用的入口帧。一个 `MAX_STEPS=1` 真实训练 smoke 已确认训练路径可用：aux target 为 `pre_first_close_pose_xyz`，prediction/label shape 都是 `(1,3)`，aux loss 为 finite `0.0459`。随后小样本 paired diagnostic 也能加载该 checkpoint 并跑通，`paired_depth_point_xyz_l2=0.1888`，`paired_pred_xyz_l2=0.00819m`。该临时 checkpoint 已删除以节省磁盘，只保留 JSON 诊断日志。

之后又从 `first_close_pose_xyz` chunk8 checkpoint 继续训练了一个 500-step `pre_first_close_pose_xyz` gate。该 no-go checkpoint 已删除以节省约 `855M` 磁盘，只保留 JSON 诊断日志：

```text
experiments/logs/pre_first_close_scale20_chunk8_500_paired/
experiments/logs/pre_first_close_scale20_chunk8_500_strict/
```

paired normal-vs-cross 仍然有明显几何/action 差异：

| metric | value |
|---|---:|
| `paired_depth_point_xyz_l2` | `0.1686` |
| `paired_depth_waypoint_xyz_action_l2` | `0.5694` |
| `paired_depth_waypoint_chunk_xyz_action_l2` | `1.6105` |
| `paired_pred_xyz_l2` | `0.00690m` |

strict normal/null/cross 结果：

| depth mode | `xyz_rmse` | `xyz_direction_cosine` | `pred_xyz_norm` |
|---|---:|---:|---:|
| normal | `0.00382` | `0.588` | `0.00802` |
| null | `0.00491` | `0.251` | `0.00665` |
| cross_sample | `0.00357` | `0.753` | `0.00831` |

解读：`pre_first_close_pose_xyz` 比 generic visible point 更贴近 demo-tail 成功接管入口，也让 normal 明显优于 null；但 cross-sample 仍然在 strict imitation 指标上优于 normal，所以不能写成 learned OpenVLA RGB-D success。当前最稳妥的 OpenVLA 结论仍是：depth 已进入 temporal action geometry，但 object/contact-specific grounding 和后续 temporal/contact action decoder 还没有解决。

最后再把 target 从“demo pre-close EE xyz”改成“当前 RGB-D 可见点云中最接近 demo pre-close xyz 的点”，即 `visible_pre_first_close_point_xyz`。这个 target 更直接地要求模型从当前 depth 里选出接触前几何点，而不是只拟合一个轨迹坐标。

代码和诊断：

```text
runs_rlbench_visible_preclose_500/
experiments/logs/rlbench_visible_contact_target_visible_preclose_probe.json
experiments/logs/visible_preclose_500_paired/
experiments/logs/visible_preclose_500_strict/
```

target feasibility probe 先证明这个 label 本身是 depth-sensitive 的：

| target | normal median | cross median | EE fallback median |
|---|---:|---:|---:|
| `pre_first_close_pose` | `0.0615m` | `0.0508m` | `0.1084m` |
| `visible_pre_first_close_point` | `0.0000m` | `0.0142m` | `0.0934m` |

500-step 训练也跑通：`MAX_STEPS=1` smoke 中 aux target 为 `visible_pre_first_close_point_xyz`，prediction/label shape 都是 `(1,3)`，aux loss `0.0405`；500-step 训练第一步 aux loss 为 `0.0315`。

最重要的是 selected-point 几何误差通过了 normal/null/cross gate：

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
| normal advantage | `0.095m` |
| `paired_depth_point_xyz_l2` | `0.1746m` |
| `paired_depth_waypoint_chunk_xyz_action_l2` | `1.8790` |
| `paired_pred_xyz_l2` | `0.00818m` |

解读：这仍然不是 rollout success，因为 strict action imitation 里 cross-sample 的 `xyz_rmse` 和 cosine 仍优于 normal；但它已经给出了 OpenVLA 端到端几何层面的正证据：真实 normal depth 让模型选出的 3D 点更接近当前可见 pre-contact label，并且这个点继续写入 8-step waypoint/action chunk。当前剩余问题不再是“depth 没进入几何信号”，而是“几何点到可执行 contact trajectory 的 action decoder 仍不够”。

已做验证：真实 `open_drawer` HDF5 上三种 target 都能生成 finite `(3,)` 标签，runner dry-run 能正确传入 `--aux_future_horizon`；`MAX_STEPS=1` 真实训练 smoke 也通过，`farthest_future_pose_xyz` 的 prediction/label shape 均为 `(1, 3)`。

随后补跑了一个 500-step `farthest_future_pose_xyz` 门槛实验。结果仍然是 NO-GO：

| 检查项 | 结果 |
|---|---:|
| paired normal-vs-cross `paired_pred_xyz_l2` | `1.66e-04` |
| strict normal `xyz_rmse` | `0.003190` |
| strict null `xyz_rmse` | `0.003210` |
| strict cross-sample `xyz_rmse` | `0.003189` |

随后又补跑了一个更完整的 5000-step `farthest_future_pose_xyz` gate。结果仍然是 NO-GO，而且 paired depth effect 进一步变小：

| 检查项 | 结果 |
|---|---:|
| paired normal-vs-cross `paired_pred_l1` | `1.58e-05` |
| paired normal-vs-cross `paired_pred_rmse` | `3.79e-05` |
| paired normal-vs-cross `paired_pred_xyz_l2` | `1.00e-04` |
| strict normal `xyz_rmse` | `0.003167` |
| strict null `xyz_rmse` | `0.003210` |
| strict cross-sample `xyz_rmse` | `0.003162` |

这个结果说明：只把 auxiliary target 换成长视野 waypoint，甚至训练到 `5000` steps，仍然不能让当前 OpenVLA-OFT action path 因果依赖真实 depth。下一步必须换成更彻底的 3D action-map / diffusion-style action decoder，而不是继续扩大同一 waypoint recipe。

这些 no-go checkpoints 已删除以节省磁盘，只保留 JSON 日志和复现命令。

## 7. 最后探索：ManiSkill3 point-cloud decoder

因为 clean LIBERO 已接近饱和，而 RLBench/OpenVLA-OFT residual/waypoint recipe 已经连续 no-go，提交前又做了一个最后的路线探索：把 ManiSkill3 官方 demo replay 成 pointcloud observation，并训练 primary point-cloud / object-centric action decoder。

早期 20-demo pilot 还不是最终 positive result，因为当时 learned policy 的 closed-loop rollout 仍然失败，也没有 matched no-depth rollout baseline；但它给出了下一步方向的强信号。

数据与设置：

| task | demos | transitions | obs | action |
|---|---:|---:|---|---|
| `PushCube-v1` | `20` | `1371` | pointcloud | `pd_ee_delta_pos`, dim `4` |
| `PickCube-v1` | `20` | `1493` | pointcloud | `pd_ee_delta_pos`, dim `4` |

关键结果：

| task | pointcloud normal RMSE mean | null RMSE mean | cross RMSE mean | paired normal-vs-cross L2 mean | gate |
|---|---:|---:|---:|---:|---:|
| `PushCube-v1` | `0.015634` | `0.020435` | `0.015687` | `0.002041` | `3/3` seeds |
| `PickCube-v1` | `0.119985` | `0.140440` | `0.120763` | `0.022263` | `2/3` seeds |

对比 RLBench 最后 waypoint recipe 的 paired normal-vs-cross `paired_pred_xyz_l2=1.00e-04`，PickCube 的 pointcloud decoder 已经出现大一个数量级以上的 action sensitivity。这个早期 offline pilot 还不能单独写成 RGB-D success；但它说明下一步最合理的路线是 ManiSkill3/DP3-style primary point-cloud action decoder，而不是继续在 LIBERO 或 OpenVLA-OFT optional residual 上调参。

随后又补了一个 PickCube closed-loop smoke。结果仍然没有成功：

| checkpoint | mode | success | mean reward |
|---|---|---:|---:|
| pointcloud 1200-step | normal | `0/3` | `7.09` |
| pointcloud 1200-step | null | `0/3` | `4.32` |
| pointcloud 1200-step | cross_demo | `0/3` | `7.33` |
| proprio 1200-step | null | `0/3` | `6.81` |
| pointcloud 5000-step | normal | `0/3` | `10.31` |
| pointcloud 5000-step | null | `0/3` | `2.91` |
| pointcloud 5000-step | cross_demo | `0/3` | `10.75` |

解读：离线 action sensitivity 可以放大，但 tiny single-step PointNet BC decoder 仍然不能完成闭环控制。下一步应换成 chunked/diffusion action decoder 或更完整的 3D action-map policy。

进一步补做了一个 action chunk 版本，预测未来 `8` 步 action，rollout 时执行前 `4` 步：

| setting | success | mean reward |
|---|---:|---:|
| chunk normal, execute 4 | `0/3` | `5.46` |
| chunk null, execute 4 | `0/3` | `3.01` |
| chunk cross_demo, execute 4 | `0/3` | `5.46` |
| chunk normal, execute 1 | `0/3` | `3.57` |

离线 chunk gate 仍然通过，paired normal-vs-cross step L2 为 `0.027554`；但闭环仍失败，而且 normal 与 cross_demo 几乎一样。因此简单 action chunk 也不是最终解法，后续需要真正的 temporal policy，例如 DP3、diffusion action decoder 或 ACT-style temporal aggregation。

再补了 goal-conditioned PointNet，把 `goal_pos` 加入输入。它仍然没有通过 strict offline gate：normal RMSE `0.196182`，null `0.206774`，cross_sample `0.195966`，paired normal-vs-cross L2 为 `0.019890`。闭环 normal/null/cross 均为 `0/3`，且 cross_demo reward 最高。因此只加 goal conditioning 不够。

最后补了 object-centric feature MLP：用 pointcloud segmentation 提取 cube center，再拼接 `tcp_pose`、`goal_pos`、`is_grasped`、相对 3D 向量和 proprio。这个版本强通过离线 causal gate：

| mode | raw RMSE |
|---|---:|
| normal | `0.130584` |
| null | `51700.929688` |
| cross_sample | `0.990587` |

paired normal-vs-cross L2 达到 `1.314387`。但闭环仍然全部失败：

注意：object-feature 的 null 是明显 OOD，因为训练时 `cube_valid` 基本恒为 `1`，而 null 把它置为 `0`；因此这组更可信的 causal 对比是 normal vs cross_sample。

| mode | success | mean reward |
|---|---:|---:|
| normal | `0/10` | `4.83` |
| null | `0/10` | `0.51` |
| cross_demo | `0/10` | `5.25` |

解读：object-centric features 已经能让 action prediction 强依赖真实几何，但单步 BC 仍然无法闭环控制。失败点从“depth 是否有信号”进一步收紧到“temporal policy / compounding error”。

最后又补了一个显式 pointcloud geometry controller，用 pointcloud segmentation 估计 cube 位置，再结合 `tcp_pose` / `goal_pos` 做几何状态机控制。这个不是 learned VLA policy，但它回答了“当前 PickCube 是否真的可由 RGB-D/pointcloud 几何闭环解决”：

| mode | success | mean reward |
|---|---:|---:|
| normal pointcloud | `7/10` | `26.38` |
| null pointcloud | `0/10` | `5.84` |
| cross_demo pointcloud | `0/10` | `9.23` |

在 150-step horizon 和 last-cube memory 下：

| mode | success | mean reward |
|---|---:|---:|
| normal pointcloud | `8/10` | `34.70` |
| null pointcloud | `0/10` | `8.76` |
| cross_demo pointcloud | `1/10` | `17.66` |

随后用 geometry controller 生成成功 teacher rollouts，并训练 object-feature policy。无 phase 版本显示 normal reward 很高但仍不成功：normal `0/10`，mean reward `54.56`；null `0/10`，`3.67`；cross-demo `0/10`，`12.83`。Debug 显示它能抓住 cube，但不会稳定切到 move-goal 阶段。

最后加入显式 phase one-hot，由几何状态机提供 phase，但 action 仍由 learned MLP 输出。这个 phase-conditioned teacher-distilled policy 先取得了一个强 learned positive diagnostic：

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

进一步解耦 action geometry 和 phase source 后，相同 seed 的 10-episode 对照为：

| action geometry | phase source | success | mean reward |
|---|---|---:|---:|
| normal | normal | `6/10` | `39.23` |
| null | normal | `0/10` | `4.47` |
| cross_demo | normal | `0/10` | `14.28` |
| normal | null | `0/10` | `14.06` |

这说明成功不是只靠 phase 提示，必须同时有真实 object geometry 和正确 temporal phase。

为了减少手写 phase 的成分，又训练了一个单帧 phase classifier，从同一套 object-centric 3D features 预测 phase。这个 learned phase classifier 的 validation accuracy 为 `96.1%`，macro accuracy 为 `91.6%`。在 rollout 中用预测 phase 替代状态机 phase 后，30-episode 结果为：

| eval | normal | null | cross_demo |
|---|---:|---:|---:|
| learned phase, 30 episodes | `19/30` | `0/30` | `0/30` |

进一步解耦 learned phase source 和 action geometry source：

| action geometry | learned phase source | success | mean reward |
|---|---|---:|---:|
| null | normal | `0/10` | `4.19` |
| cross_demo | normal | `0/10` | `11.99` |
| normal | null | `0/10` | `15.11` |

这个 result 很重要：真实点云几何不仅可以被显式 controller 使用，也可以通过 learned action decoder 转成闭环成功，并且 normal 明显超过 null/cross controls。限制也必须讲清楚：这还不是完整端到端 VLA，因为输入仍是 segmentation-derived object features，而不是 raw RGB-D tokens；下一步应让 ACT/DP3/recurrent/diffusion policy 学习 temporal state，并把 object-centric geometry 接到更强的 action decoder。

最后又把输入往 raw pointcloud 推了一步：不再把 `cube_center` 作为输入，而是用 `z>0.02` cropped pointcloud xyz/rgb，加上 tcp/goal/grasped/proprio task state，训练 PointNet trunk 同时预测 phase、cube-center auxiliary 和 action。

离线结果强通过 normal/null/cross gate：

| mode | action RMSE | phase acc | cube RMSE |
|---|---:|---:|---:|
| normal | `0.103` | `94.3%` | `0.009m` |
| null | `0.184` | `89.5%` | `0.076m` |
| cross_sample | `0.215` | `87.7%` | `0.075m` |

paired normal-vs-cross action L2 为 `0.215`。闭环 30 episodes 得到一个弱正信号：

| eval | success |
|---|---:|
| raw cropped pointcloud, 30 teacher episodes, normal/null/cross | `2/30` / `0/30` / `0/30` |
| raw cropped pointcloud, 100 teacher episodes, seed4100, normal/null/cross | `8/30` / `1/30` / `1/30` |
| raw cropped pointcloud, 100 teacher episodes, seed4500, normal/null/cross | `12/30` / `0/30` / `0/30` |
| raw cropped pointcloud, 100 teacher episodes, aggregate, normal/null/cross | `20/60` / `1/60` / `1/60` |
| matched sampled-RGB-only train baseline | `1/60` |
| matched null/proprio train baseline | `3/60` |

这比 object-feature learned-phase 的 `19/30` 更接近 raw RGB-D/pointcloud 输入，而且已经不只是弱信号：扩大 teacher 数据和模型容量后，normal pointcloud 在两个独立 eval seed 上都明显超过 null/cross controls，也超过训练时无 3D 几何的 matched baselines。边界是：这是 ManiSkill teacher-distilled PointNet policy，不是 OpenVLA 端到端；但这个边界不等于“OpenVLA RGB-D 无效”，它说明 crop + cube auxiliary + primary pointcloud action decoder 是有效方向。

为了切开 perception 和 action decoder，又把同一个 raw pointcloud cube predictor 接到固定 geometry controller。这个不是 learned action policy，但它直接测试 learned raw pointcloud perception 是否足够用于控制：

| eval | normal | null | cross_demo |
|---|---:|---:|---:|
| learned cube + fixed controller | `22/30` | `1/30` | `0/30` |

这说明 raw cropped pointcloud 的 learned perception 已经很强；真正没有学好的，是从几何/phase 到连续动作的 temporal action decoder。

详细记录见：

```text
MANISKILL_FINAL_PILOT.md
experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_seed7_10k_h256.json
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_normal_seed7_150steps_30eps.json
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_normal_seed7_150steps_seed4500_30eps.json
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_null_seed7_150steps_30eps.json
experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_cross_demo_seed7_150steps_30eps.json
experiments/logs/maniskill_pickcube_rollout_learned_cube_controller_cropz002_normal_seed7_150steps_30eps.json
```

## 8. 最终汇报一句话

> 我们证明了 depth/pointcloud 在更 3D-sensitive 的 PickCube 任务上能转成 learned policy 的闭环收益：raw cropped pointcloud policy 扩大到 `100` 条成功 teacher 轨迹后，normal 达到 `20/60`，eval-time null/cross 只有 `1/60`，训练时 matched RGB-only/null baselines 也只有 `1/60` 和 `3/60`。

补充一句：

> OpenVLA-OFT/LIBERO 的轻量 adapter 路线没有形成正结果，但这个 no-go 对照正好说明：depth 不能只作为可选 residual 加进去，必须放到更 3D-sensitive 的数据集和 primary pointcloud/action decoder 里。这不是证明 OpenVLA RGB-D 没用，而是说明当前证据支持先换 benchmark/action decoder。

## 9. 证据位置

- final results table: `FINAL_RESULTS_TABLE.md`, `experiments/logs/final_results_table.csv`
- final evidence index: `FINAL_EVIDENCE_INDEX.md`
- OpenVLA visible-precontact gate: `experiments/logs/visible_preclose_500_paired/`, `experiments/logs/visible_preclose_500_strict/`
- ManiSkill3 raw pointcloud main result: `experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_seed7_10k_h256.json`
- ManiSkill3 normal/null/cross rollouts: `experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_*_30eps.json`
- matched no-depth baselines: `experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_rgbonly_seed7_10k_h256.json`, `experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_nulltrain_seed7_10k_h256.json`
- learned cube + fixed controller: `experiments/logs/maniskill_pickcube_rollout_learned_cube_controller_cropz002_*_150steps_30eps.json`
- loop records: `.loop/state.md`, `.loop/failures.md`, `.loop/decisions.md`
