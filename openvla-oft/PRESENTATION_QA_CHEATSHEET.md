# DepthVLA-OFT 答辩问答速查

更新时间：2026-07-05 UTC

## 0. 总口径

如果老师问一个很宽的问题，先用这一版：

> 我们最后证明的是：depth/pointcloud 在更 3D-sensitive 的 ManiSkill3 PickCube 上是有因果作用的。把数据扩到 `100` 条成功 teacher 轨迹后，raw cropped pointcloud learned policy 在两组 eval seed 上达到 normal `20/60`，而 eval-time null/cross-demo 都只有 `1/60`；训练时 matched sampled-RGB-only baseline 是 `1/60`，matched null/proprio baseline 是 `3/60`。这说明深度几何不是没用，关键是要换掉 LIBERO 这种接近饱和的 benchmark，并把 pointcloud 接到 primary action decoder，而不是作为 OpenVLA-OFT 的可选 residual。

注意边界，不要把 ManiSkill3 结果包装成：

- OpenVLA-OFT 端到端 RGB-D 已经超过 RGB-only。
- OpenVLA waypoint/action 已经取得任务成功。
- 只要换更大数据集就一定能赢。

更好的说法：

> 我们这轮没有把 OpenVLA-OFT 端到端 RGB-D 作为主正证据，但这不等于证明 OpenVLA RGB-D 没用。真正被证明的是：在 ManiSkill3 PickCube 这种更 3D-sensitive 的设置里，pointcloud 几何进入 primary learned action decoder 后，normal 明显超过 matched no-depth baselines。

## 1. 最重要的正结果是什么？

答：

> 最重要的正结果是 ManiSkill3 PickCube 上的 raw cropped pointcloud teacher policy。它不输入 ground-truth cube center，只输入 `z>0.02` cropped pointcloud xyz/rgb 加 task state，并且只在训练时用 cube center 做 auxiliary supervision。扩大到 `100` 条成功 teacher 轨迹后，normal pointcloud 达到 `20/60`，eval-time null/cross-demo 都是 `1/60`；同模型容量、同数据、同 eval seeds 的 sampled-RGB-only train baseline 也是 `1/60`，null/proprio train baseline 是 `3/60`。这是 learned policy 的闭环差距，不只是 offline probe。

可以补充：

- learned-phase object-feature policy 更强：normal `19/30`，null/cross `0/30`。
- learned cube + fixed controller 是 perception/action 拆分诊断：normal `22/30`，null `1/30`，cross `0/30`。
- OpenVLA-OFT/LIBERO 部分不是主正证据，而是说明 saturated benchmark 和 optional residual fusion 为什么会掩盖 depth；它不是“OpenVLA RGB-D 无效”的证明。

## 2. 最后一次关键实验是什么？

答：

> 如果按主正结果说，最后关键实验是 ManiSkill3 PickCube 的 `100` 条成功 teacher 轨迹 raw cropped pointcloud policy：normal `20/60`，eval-time null/cross-demo 各 `1/60`，matched sampled-RGB-only/null-train baselines 是 `1/60` 和 `3/60`。这是真正能写进最终 claim 的 learned closed-loop depth gain。

如果老师追问 OpenVLA/RLBench 的最后一次尝试：

> 最后一次 OpenVLA 尝试是 `visible_pre_first_close_point_xyz + 8-step waypoint action chunk`。它把 demo pre-contact 入口投到当前 RGB-D 可见点云上，让模型从 dense point tokens 里选当前可见的 3D 点，而不是只拟合一个抽象 EE 坐标。

结果：

| 指标 | 数值 |
|---|---:|
| normal selected point -> aux label | `0.099m` |
| null selected point -> aux label | `0.699m` |
| cross selected point -> aux label | `0.194m` |
| paired selected-point normal advantage | `0.095m` |
| paired waypoint chunk action L2 | `1.8790` |
| strict action imitation | cross-sample 仍优于 normal |

一句话：

> OpenVLA 这里已经能证明 depth 进入了几何 bottleneck 和 action chunk，但还不能证明 rollout success 或 RGB-D 超过 RGB-only；主成功证据仍然放在 ManiSkill3 raw pointcloud policy。

## 3. 为什么不继续用 LIBERO？

答：

> 因为 clean LIBERO 对强 RGB-only OpenVLA-OFT 已经接近饱和。我们这里历史结果能到 clean trained tasks `15/15`，这种天花板会掩盖 depth 的边际收益。LIBERO 适合 sanity check，但不适合作为证明 depth value 的主 benchmark。

如果追问换什么：

> 第一优先是 RLBench，因为它有 RGB-D、相机参数、closed-loop eval 和大量 3D manipulation tasks。ManiSkill3 适合后续高吞吐 synthetic RGB-D/point-cloud 数据，RoboCasa365 和 RoboTwin 2.0 更适合第二阶段大规模泛化。

## 4. 为什么 RLBench 也没成功？

答：

> RLBench 解决了 benchmark 饱和问题，但没有自动解决 action-space 问题。`open_drawer` 上 depth 的离线空间信号存在，可是当前 OpenVLA-OFT action path 仍然可以绕开 depth，或者只学到很小、不稳定的 depth perturbation。

重点：

- 换数据集是必要条件，不是充分条件。
- depth 必须进入 action 表示本身，例如 voxel/3D action map、coarse-to-fine keypose detection、diffusion action decoder。
- 最新 visible-precontact gate 已经说明 depth 能进入 selected 3D point 和 action chunk；失败边界后移到了 contact pose、gripper orientation 和 post-contact temporal trajectory。

## 5. 怎么证明 depth 数据不是坏的？

答：

> projected keypose heatmap probe 显示 normal depth 明显好于 corrupt depth。比如 `open_drawer` normal peak error `2.85px`，cross-sample `8.86px`，null `43.30px`；stable6 normal `2.94px`，cross-sample `12.53px`，null `57.63px`。

所以：

> 原始 depth 有 spatial signal，失败发生在把 signal 接入 final action 的阶段。

## 6. 为什么 normal/null/cross-sample 消融很重要？

答：

> RGB-D 成功率本身不能证明模型用了 depth。只有 normal 明显优于 null 和 cross-sample，才能说明 depth 内容对 action 有因果贡献。cross-sample 比 pixel shuffle 更强，因为 dense point tokens 可能保留很多统计结构，简单 shuffle 不一定破坏几何 shortcut。

## 7. 为什么最后不跑更大训练量？

答：

> 因为这条 OpenVLA/RLBench selected-point/waypoint recipe 没有过完整 action/rollout gate。最新 visible-precontact target 的 selected-point geometry 过了 normal/null/cross gate，但 strict action imitation 里 cross-sample 仍优于 normal，也没有 rollout success。继续扩大同一个 recipe，很可能只是在优化几何点选择，而不是学会 contact-level temporal action。

可以承认：

> 更大数据量可能有帮助，但在没有通过 normal/null/cross gate 前，它不能作为正结果的证据。

## 8. 为什么不是实现 bug？

答：

> 我们确实发现并修了一个关键 bug：RLBench intrinsics 里 `fx/fy` 可能为负，旧 backprojection 把负 focal length clamp 到 `1e-6`，导致 point cloud 爆到异常尺度。修复后 point cloud 回到米级，离线 depth probe 能正常区分 normal/null/cross。

所以更合理的判断是：

> 已知数据和几何管线不是完全坏的；问题主要在 policy action formation。

## 9. 相关工作给了什么启发？

答：

> PerAct 把 RGB-D observation 和 voxelized 6-DoF action space 绑定；Act3D 把 6-DoF keypose prediction 当作 3D detection；DP3 用 sparse point cloud 直接条件化 diffusion action sequence；BridgeVLA 把 3D point cloud 投影成多视角 2D，并用 heatmap 做 input-output alignment。这些都说明 depth 不能只是附加 feature，而要和 action space 对齐。

对应到我们的结果：

> 我们尝试了 residual、heatmap、point selection、waypoint patch，但还没做到完整 3D action-space redesign。

## 10. 下一步最合理怎么做？

答：

> 如果继续做，我会停止在 OpenVLA-OFT action head 上打小补丁，改成完整 3D action decoder：Act3D/PerAct-style 3D action map，或者 DP3-style point-cloud diffusion action head。更重要的是，target 不能再只是接近当前 EE 的 next pose/keypose，而要改成 object/contact-conditioned waypoint 或 task-level 3D action target。数据上优先 RLBench/ManiSkill3，小规模先过 normal/null/cross causal gate，再扩大到更多任务和 demonstrations。

Go/no-go：

- normal 必须明显优于 null/cross。
- normal 必须优于 matched RGB-only。
- 提升必须出现在 3D/contact/viewpoint-sensitive tasks，而不是只靠语言或 proprio shortcut。

如果老师问“为什么 action map 也不一定行”：

> 我们做了一个 3D action-map feasibility probe。短步 keypose/next-pose target 下，current EE fallback 反而比 point candidates 更近，说明 label 太短视；但换成 future/final/farthest-future target 后，normal point candidates 明显优于 cross-sample 和 EE fallback。所以不是 action map 没希望，而是 action map 必须配更长视野、更 task-level 的 target。

## 11. 最后换 ManiSkill3 试了吗？

答：

> 试了一个很小的 offline pilot。我们用 ManiSkill3 官方 demo replay 出 pointcloud observation 和 `pd_ee_delta_pos` action，然后训练 primary point-cloud action decoder，不再把 depth 当 optional residual。

关键结果：

- `PushCube-v1`：20 demos，pointcloud gate `3/3` seeds 通过，但 normal-vs-cross L2 mean 只有 `0.002041`。
- `PickCube-v1`：20 demos，pointcloud gate `2/3` seeds 通过，normal-vs-cross L2 mean 达到 `0.022263`。
- 对比 RLBench farthest-future waypoint 的 `1e-4` 级 action delta，PickCube 明显更有希望。
- 当时闭环 smoke 仍是 `0/3`：5000-step pointcloud normal reward 高于 null，但 cross_demo 也不低，所以那一版还不能作为 normal geometry closed-loop gain 的主证据。
- 我们还试了一个 8-step action chunk，离线 normal-vs-cross step L2 是 `0.0276`，但闭环仍 `0/3`，normal 和 cross_demo reward 几乎一样。
- 又试了 goal-conditioned PointNet：normal RMSE 好于 null，但不如 cross_sample，闭环 normal/null/cross 全部 `0/3`。
- 最后试了 object-centric feature MLP：用 segmentation 提取 cube center 后，离线 gate 强通过，paired normal-vs-cross L2 达到 `1.31`；但闭环 normal/null/cross 仍全部 `0/10`。
- 最后做了 pointcloud geometry controller：100-step normal `7/10`，null/cross `0/10`；150-step normal `8/10`，null `0/10`，cross `1/10`。这说明 PickCube 确实能被真实点云几何解决，学不到是 policy 问题。
- 再最后做了 geometry-teacher distillation：无 phase 版本 normal reward 很高但 success 仍 `0/10`；加入 phase one-hot 后，learned MLP action decoder 在 30 episodes 上 normal `17/30`，null `0/30`，cross-demo `0/30`。
- 为了排除“只是 phase 起作用”，又做了 phase/geometry 解耦：normal geometry + normal phase 是 `6/10`；null geometry + normal phase、cross geometry + normal phase、normal geometry + null phase 都是 `0/10`。
- 最后把手写 phase 替换成 learned phase classifier：classifier val accuracy `96.1%`，rollout normal `19/30`，null `0/30`，cross-demo `0/30`。
- learned-phase 解耦也过了：null geometry + learned normal phase、cross geometry + learned normal phase、normal geometry + learned null phase 都是 `0/10`。
- 又补了 raw cropped pointcloud follow-up：不把 `cube_center` 作为输入，只用 `z>0.02` cropped pointcloud + task state，离线 normal action RMSE `0.103`，cross `0.215`，cube RMSE normal `0.009m`、cross/null 约 `0.075m`；闭环 action head normal `2/30`，null/cross `0/30`。
- 最后扩大 raw pointcloud teacher 数据到 `100` 条成功轨迹，训练 h256/10k 单步 action decoder：离线 normal action RMSE `0.067`，null `0.210`，cross `0.131`；两组 30-episode eval seed 合计 normal `20/60`，null `1/60`，cross-demo `1/60`。
- matched no-depth 训练 baseline 也补了：sampled-RGB-only train baseline `1/60`，null/proprio train baseline `3/60`。
- 同样数据的 action chunk decoder 离线也过 gate，但闭环只有 normal `2/30`、null/cross `1/30`，说明最后的提升主要来自更多 teacher data 和更稳的 primary pointcloud action decoder，不是简单 chunking。
- 把同一个 learned cube predictor 接到固定 geometry controller 后，normal `22/30`，null `1/30`，cross `0/30`，说明 raw pointcloud perception 已经够强，瓶颈是 learned action/temporal decoder。

但要强调：

> 这不是把 OpenVLA 判成没用，而是说明最终正证据来自另一条更适合 depth 的路线：ManiSkill3/DP3-style primary pointcloud/temporal policy，而不是 LIBERO 或 OpenVLA residual 调参。object-centric MLP 说明几何可以被接入 action prediction；geometry controller 说明几何可以闭环成功；learned-phase distillation 说明 learned action decoder 也可以用 normal geometry 成功控制。剩下的 gap 是把这些能力进一步迁移到更端到端的 raw RGB-D/pointcloud temporal policy。

如果老师问“这算不算正结果？”：

> 现在 phase 已经可以由 learned classifier 预测，所以“手写 phase”不再是唯一解释；但输入仍然是 segmentation-derived object features。它是 learned positive diagnostic，说明 geometry 可以被策略使用；边界是它还不是最终端到端 VLA 正结果。

如果老师问“有没有更接近 raw RGB-D 的结果？”：

> 有，而且最后已经比弱正更明确了：我们不再输入 cube center，而是输入 cropped raw pointcloud，并只用 cube center 做训练时 auxiliary supervision。30 条 teacher 轨迹时是 normal `2/30`、null/cross `0/30`；扩大到 100 条成功 teacher 轨迹后，两组 eval seed 合计 normal `20/60`、eval-time null/cross-demo 各 `1/60`，matched sampled-RGB-only train baseline `1/60`，matched null/proprio train baseline `3/60`。这说明 raw pointcloud 里的几何确实能通过 learned action policy 转成闭环收益。边界是：它是 ManiSkill teacher-distilled policy，不是 OpenVLA；但这不是无效性证明，而是证据来源的边界。

如果老师追问“raw pointcloud perception 到底够不够？”：

> 够到能支撑控制。我们把 learned cube predictor 接到固定 geometry controller，normal 达到 `22/30`，null `1/30`，cross `0/30`。这说明 raw depth/pointcloud 里的物体几何已经被模型学出来了；失败主要在 learned action decoder，而不是 perception。

## 12. 如果老师问“那你项目到底贡献是什么？”

答：

> 贡献分两层：第一，我们构建了 RGB-D OpenVLA-OFT/RLBench 管线，加入 normal/null/cross-sample causal ablation，验证了 depth 有离线空间信号，修复了关键 point-cloud bug，并系统排除了多种轻量 fusion 路线；第二，我们在 ManiSkill3 PickCube 上给出了正向闭环证据，证明 pointcloud 几何进入 primary learned action decoder 后可以明显超过 no-depth/corrupt-depth controls。最终结论不是“OpenVLA RGB-D 没用”，而是 depth 必须和 3D-sensitive benchmark、object-centric representation、primary/temporal action decoder 绑定。

## 13. 最短现场版

- OpenVLA/RLBench recipe 为什么 no-go？
  - 最新几何 bottleneck 过了 selected-point gate，但没有过 strict action/rollout gate。
- depth 数据有用吗？
  - 有，离线 heatmap/keypose probe 证明 normal depth 好于 null/cross。
- 为什么不把它说成 OpenVLA 成功？
  - OpenVLA/RLBench 还没有 rollout success，也没有证明超过 RGB-only；主成功证据来自 ManiSkill3 raw pointcloud policy。
- 为什么换出 LIBERO？
  - clean LIBERO RGB-only 接近天花板。
- 最后实验是什么？
  - 主正结果是 ManiSkill3 PickCube `100` 条 teacher 的 raw pointcloud policy；OpenVLA 最后是 visible-precontact point gate。
- 最后结果？
  - ManiSkill normal `20/60`，null/cross `1/60`，matched RGB-only/null train `1/60`/`3/60`；OpenVLA 只证明 depth 进入几何/action chunk。
- 还有什么新诊断？
  - 短步 target 有 EE shortcut；future/final/farthest target 更能体现 depth 价值。
- 最后 ManiSkill 试验说明什么？
  - learned-phase object-feature policy normal `19/30`，null/cross `0/30`；scaled raw cropped pointcloud action head normal `20/60`，eval-time null/cross 各 `1/60`，matched RGB-only/null train baseline `1/60`/`3/60`；learned cube + fixed controller normal `22/30`，null/cross 基本为 0。
- 下一步？
  - object/contact-conditioned 3D action target + 完整 3D action map / diffusion action decoder。

## 14. 可引用来源

- 官方来源与实验决策核对：`FINAL_SOURCE_AUDIT.md`
- RLBench: https://sites.google.com/view/rlbench
- ManiSkill3 docs: https://maniskill.readthedocs.io/en/latest/user_guide/index.html
- RoboCasa365: https://robocasa.ai/
- RoboTwin 2.0: https://github.com/robotwin-Platform/robotwin
- CALVIN action space: https://github.com/mees/calvin
- PerAct: https://peract.github.io/
- Act3D: https://act3d.github.io/
- DP3: https://3d-diffusion-policy.github.io/
- BridgeVLA: https://arxiv.org/html/2506.07961v2
