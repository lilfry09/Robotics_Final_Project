# ManiSkill3 最后探索：Point-Cloud Action Decoder

更新时间：2026-07-04 UTC

## 结论

这是本项目最后最有用的正向证据：在 ManiSkill3 PickCube 这种更 3D-sensitive 的任务上，depth/pointcloud 终于转成了 learned policy 的闭环收益。

相比 RLBench/OpenVLA-OFT residual/waypoint recipe 里 `1e-4` 级的 normal-vs-cross action delta，ManiSkill3 上的 primary point-cloud / object-centric decoder 已经能产生稳定可测的 depth/pointcloud action sensitivity；显式 geometry controller 证明 PickCube 确实能被真实点云闭环解决；phase-conditioned distillation 进一步证明 learned action decoder 可以在 normal geometry 下超过 null/cross controls；learned-phase follow-up 又把手写 phase 替换成单帧 phase classifier，normal 仍达到 `19/30`，null/cross 仍为 `0/30`；最后扩大 raw cropped pointcloud teacher 数据到 `100` 条成功轨迹后，learned raw-pointcloud action policy 在两组 30-episode eval seed 上合计 normal `20/60`，eval-time null/cross-demo 各 `1/60`，matched sampled-RGB-only/null train baselines 分别只有 `1/60` 和 `3/60`。

边界也要讲清楚：主证据来自 ManiSkill teacher-distilled pointcloud policy，不是 OpenVLA matched RGB-only rollout baseline；但这个边界不是“OpenVLA RGB-D 无效”的证明。它已经足够证明“depth/pointcloud 几何有用”，并且说明下一步应该沿着 primary pointcloud/temporal action decoder 继续做。

最稳妥 claim：

> LIBERO 已饱和，RLBench residual/waypoint recipe 已 no-go；最后的 ManiSkill3 pilot 说明，换到更高吞吐 RGB-D/pointcloud 数据，并把 point cloud 接到 object-centric / temporal primary action decoder，比继续做可选 depth residual 更有希望。当前最强 learned action 结果是 learned-phase object-feature policy：normal `19/30`，null `0/30`，cross-demo `0/30`；当前最强 raw-pointcloud learned action 结果是 success100/h256/10k PointNet policy：normal `20/60`，eval-time null/cross-demo 各 `1/60`，matched sampled-RGB-only/null train baselines 为 `1/60`/`3/60`；最强 raw pointcloud perception diagnostic 是 learned cube + fixed controller：normal `22/30`，null `1/30`，cross-demo `0/30`。

## 数据

使用 ManiSkill3 官方 demo，并用官方 `replay_trajectory` 生成 pointcloud observation：

```text
PushCube-v1: 20 demos, 1371 transitions, action dim 4
PickCube-v1: 20 demos, 1493 transitions, action dim 4
control mode: pd_ee_delta_pos
obs mode: pointcloud
```

生成后的文件：

```text
/root/autodl-tmp/maniskill_data/PushCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pos.physx_cpu.h5
/root/autodl-tmp/maniskill_data/PickCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pos.physx_cpu.h5
```

验证脚本：

```bash
/root/autodl-tmp/envs/maniskill3-venv/bin/python \
  experiments/robot/maniskill/validate_maniskill_hdf5.py \
  /root/autodl-tmp/maniskill_data/PickCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pos.physx_cpu.h5
```

## 模型

最小 PointNet-style decoder：

```text
sampled pointcloud xyz/rgb
  -> point MLP + max pool
proprio qpos/qvel
  -> proprio MLP
concat
  -> action MLP
  -> pd_ee_delta_pos action
```

训练只用 normal pointcloud。验证时对同一 checkpoint 比较：

- normal pointcloud
- null pointcloud
- cross-sample pointcloud，尽量来自不同 episode

同时训练 `train_point_mode=null` 的 proprio-only baseline，避免把 zero-input OOD 当成 matched no-depth baseline。

## 结果

### PushCube-v1

Pointcloud-trained，3 seeds：

| metric | mean | min | max |
|---|---:|---:|---:|
| normal raw RMSE | `0.015634` | `0.013450` | `0.019996` |
| null raw RMSE | `0.020435` | `0.018557` | `0.022271` |
| cross raw RMSE | `0.015687` | `0.013502` | `0.020004` |
| paired normal-vs-cross L2 | `0.002041` | `0.001818` | `0.002263` |
| strict gate pass | `3/3` | | |

Proprio-only baseline，3 seeds：

| metric | mean |
|---|---:|
| null raw RMSE | `0.015943` |
| paired normal-vs-cross L2 | `0.001491` |
| gate pass | `2/3` |

解读：PushCube 是平面推方块，pointcloud normal 对 null 有稳定优势，但相对 proprio-only 的优势很小；它更适合做 pipeline smoke，不适合作为强 3D claim。

### PickCube-v1

Pointcloud-trained，3 seeds：

| metric | mean | min | max |
|---|---:|---:|---:|
| normal raw RMSE | `0.119985` | `0.111342` | `0.136501` |
| null raw RMSE | `0.140440` | `0.121580` | `0.156991` |
| cross raw RMSE | `0.120763` | `0.110002` | `0.138191` |
| paired normal-vs-cross L2 | `0.022263` | `0.021175` | `0.023696` |
| strict gate pass | `2/3` | | |

Proprio-only baseline，3 seeds：

| metric | mean |
|---|---:|
| null raw RMSE | `0.122702` |
| paired normal-vs-cross L2 | `0.011329` |
| gate pass | `0/3` |

解读：PickCube 的 normal-vs-cross action delta 比 PushCube 大约一个数量级，也远大于 RLBench waypoint recipe 的 `1e-4`。但 strict loss gate 只在 `2/3` seeds 通过，matched proprio-only baseline 也只被 pointcloud normal 小幅超过。因此它是 promising pilot，不是最终成功。

## Closed-Loop Smoke

为了确认离线 action sensitivity 是否能直接变成控制成功，又补跑了 tiny decoder 的闭环 smoke。

模型：

- `pickcube_pointcloud_seed7.pt`: 20-demo pointcloud model, 1200 steps
- `pickcube_pointcloud_seed7_5k.pt`: 20-demo pointcloud model, 5000 steps
- `pickcube_proprio_seed7.pt`: 20-demo null/proprio baseline, 1200 steps

评估：`PickCube-v1`，`pd_ee_delta_pos`，3 episodes，100 max steps。

| checkpoint | point mode | success | mean reward |
|---|---|---:|---:|
| pointcloud 1200 | normal | `0/3` | `7.09` |
| pointcloud 1200 | null | `0/3` | `4.32` |
| pointcloud 1200 | cross_demo | `0/3` | `7.33` |
| proprio 1200 | null | `0/3` | `6.81` |
| pointcloud 5000 | normal | `0/3` | `10.31` |
| pointcloud 5000 | null | `0/3` | `2.91` |
| pointcloud 5000 | cross_demo | `0/3` | `10.75` |

解读：

> 离线 gate 更强不等于闭环成功。5000-step pointcloud model 的 normal/null reward 差距变大，说明 pointcloud 确实影响控制；但 cross_demo reward 也很高且全部 `0/3`，所以还不能说真实几何带来 closed-loop gain。下一步不能停留在单步 PointNet BC，需要 action chunk、diffusion policy、DAgger-style correction 或 task-level 3D action map。

## Action Chunk Attempt

继续补了一个最小 action-chunk 版本，用当前 pointcloud/proprio 预测未来 `8` 步 `pd_ee_delta_pos` action，并在 rollout 中执行前 `4` 步再重新观测。

离线 gate：

| metric | value |
|---|---:|
| normal raw RMSE | `0.206927` |
| null raw RMSE | `0.235365` |
| cross raw RMSE | `0.207193` |
| paired normal-vs-cross step L2 | `0.027554` |
| gate | passed |

闭环 smoke：

| mode | execute steps | success | mean reward |
|---|---:|---:|---:|
| normal | `4` | `0/3` | `5.46` |
| null | `4` | `0/3` | `3.01` |
| cross_demo | `4` | `0/3` | `5.46` |
| normal | `1` | `0/3` | `3.57` |

解读：

> 简单 action chunk 没有解决闭环问题。normal 仍然明显高于 null，但和 cross_demo 几乎一样，说明当前小模型学到的是 pointcloud-conditioned action distribution，而不是足够可靠的真实几何闭环控制。下一步需要更强的 temporal policy，例如 ACT-style temporal aggregation、diffusion action decoder、DP3，或者带在线纠偏的数据收集。

## Goal-Conditioned Decoder Attempt

前面的 PointNet decoder 只看 pointcloud/proprio，缺少 `goal_pos`。因此又补了一个 goal-conditioned 版本，把 `obs/extra/goal_pos` 拼到状态输入中。

离线 gate：

| mode | raw RMSE |
|---|---:|
| normal | `0.196182` |
| null | `0.206774` |
| cross_sample | `0.195966` |

paired normal-vs-cross L2 为 `0.019890`，但 normal 没有超过 cross_sample，因此 strict gate 仍然失败。

闭环 smoke，3 episodes，100 steps：

| mode | success | mean reward |
|---|---:|---:|
| normal | `0/3` | `5.97` |
| null | `0/3` | `5.53` |
| cross_demo | `0/3` | `6.61` |

解读：

> 加 `goal_pos` 有一点帮助，normal 明显优于 null，但仍没有超过 cross_sample，闭环也没有成功。问题不只是缺 goal conditioning。

## Object-Centric Feature MLP Attempt

最后又做了一个更强的 object-centric 版本：直接用 pointcloud segmentation 提取 cube center，并把 `cube_center`、`tcp_pose`、`goal_pos`、`is_grasped`、相对向量和 proprio 拼成 3D feature，再训练 MLP 预测 `pd_ee_delta_pos`。

这一步更接近“object-conditioned depth fusion”，但仍然只是小型 BC policy，不是 OpenVLA。

离线 gate：

| mode | raw RMSE |
|---|---:|
| normal | `0.130584` |
| null | `51700.929688` |
| cross_sample | `0.990587` |

| metric | value |
|---|---:|
| paired normal-vs-cross L2 | `1.314387` |
| strict gate | passed |

注意：这里 null RMSE 极大，主要是因为 object-feature 训练时 `cube_valid` 基本恒为 `1`，而 null 把它置为 `0`，属于明显 OOD；因此更可信的 causal 对比是 normal vs cross_sample。

闭环 smoke，10 episodes，100 steps：

| mode | success | mean reward |
|---|---:|---:|
| normal | `0/10` | `4.83` |
| null | `0/10` | `0.51` |
| cross_demo | `0/10` | `5.25` |

解读：

> Object-centric features 让离线 normal/null/cross 差异非常明显，说明点云几何已经被接入 action prediction；但单步 BC 仍然闭环失败。失败点进一步从“perception/action coupling”收紧到“temporal control / compounding error”。下一步应该是 DP3、diffusion policy、ACT temporal aggregation，或用 geometry controller 生成更多成功 rollouts 后再 imitation。

## Geometry-Teacher Distillation Attempt

为了验证“是不是官方 motion-planning demo 分布不适合学这个控制律”，最后又用 geometry controller 生成成功 teacher rollouts，再训练 object-feature MLP。

### 无 phase 的 teacher distillation

数据：

```text
30 successful teacher episodes
31 attempts
2538 transitions
```

训练后 validation RMSE：

| split | raw RMSE |
|---|---:|
| train | `0.015025` |
| val | `0.100967` |

闭环，150 steps，10 episodes：

| mode | success | mean reward |
|---|---:|---:|
| normal | `0/10` | `54.56` |
| null | `0/10` | `3.67` |
| cross_demo | `0/10` | `12.83` |

300-step normal 仍是 `0/10`，但 mean reward 到 `112.72`。debug 显示模型已经能抓住 cube，但一直举在高处，没有稳定切到 move-goal 阶段。这说明主要缺的是隐式 phase / temporal state。

### Phase-conditioned teacher distillation

因此又加入一个显式 phase one-hot，由同一个几何状态机维护 phase，但 action 仍由 learned MLP 输出。这不是端到端策略；它是用来验证“如果 temporal state 可用，learned low-level action decoder 是否能用真实几何成功控制”。

数据：

```text
30 successful teacher episodes
31 attempts
2335 transitions
features = object-centric 3D features + phase one-hot
```

训练后 validation RMSE：

| split | raw RMSE |
|---|---:|
| train | `0.003425` |
| val | `0.024559` |

闭环，150 steps：

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

为了排除“只是 phase 提示带来成功”的混淆，又把 action feature 的 geometry 来源和 phase 状态机的 geometry 来源解耦。相同 seed 下 10 episodes：

| action geometry | phase source | success | mean reward |
|---|---|---:|---:|
| normal | normal | `6/10` | `39.23` |
| null | normal | `0/10` | `4.47` |
| cross_demo | normal | `0/10` | `14.28` |
| normal | null | `0/10` | `14.06` |

这个对照说明：phase alone 不够，真实 cube geometry alone 也不够；成功需要 action decoder 同时看到真实 object geometry 和正确 temporal phase。

解读：

> 这是 learned positive diagnostic 的第一阶段。它的边界是 phase 由手写状态机提供，还不是 OpenVLA 端到端；但它证明了三个关键点：第一，ManiSkill3 PickCube 是合适的 3D-sensitive benchmark；第二，normal pointcloud geometry 可以让 learned action decoder 明显超过 null/cross controls；第三，之前单步 BC 失败的关键是 temporal state/phase learning，而不是 geometry 本身无效。下一节继续把 phase one-hot 换成 learned classifier，进一步减少手写状态机成分。

### Learned phase follow-up

为了减少“手写状态机”的成分，又训练了一个单帧 phase classifier：输入为同一套 object-centric 3D features，输出 `approach / descend / close / lift / move_goal / hold_goal / no_cube` phase。它不再在 rollout 时调用 phase 状态机，而是每一步根据当前 feature 预测 phase one-hot，再喂给前面的 action decoder。

离线 phase classifier：

| metric | value |
|---|---:|
| val accuracy | `96.07%` |
| val macro accuracy | `91.64%` |
| val mean confidence | `0.889` |

闭环，150 steps，30 episodes：

| mode | success | mean reward |
|---|---:|---:|
| normal | `19/30` | `32.12` |
| null | `0/30` | `3.91` |
| cross_demo | `0/30` | `11.28` |

进一步解耦 learned phase 的 geometry source 和 action feature 的 geometry source，10 episodes：

| action geometry | learned phase source | success | mean reward |
|---|---|---:|---:|
| null | normal | `0/10` | `4.19` |
| cross_demo | normal | `0/10` | `11.99` |
| normal | null | `0/10` | `15.11` |

解读：

> 这一步比手写 phase 版本更接近“可学习策略”：phase 不再由规则状态机直接提供，而是由 learned classifier 预测。normal `19/30`、null/cross `0/30` 说明真实 object geometry 仍然是因果必要信息；解耦结果也说明 learned phase 不能单独救场。限制仍然必须讲清楚：输入还是 segmentation-derived object features，不是 raw RGB-D/OpenVLA；phase classifier 是单帧模型，不是真正的历史策略。下一步应该把 object-centric geometry 接入 ACT/DP3/recurrent/diffusion policy，让 temporal state 由策略内部学习。

### Raw cropped pointcloud follow-up

为了把输入从 `cube_center` feature 往 raw pointcloud 推一步，又补做了一个 raw pointcloud teacher-distillation 版本。输入不再包含 segmentation-derived cube center，而是：

```text
sampled pointcloud xyz/rgb
  + tcp/goal/grasped/proprio task state
  -> PointNet trunk
  -> phase logits + cube-center auxiliary head + action head
```

关键实现细节：

- 采样前做 `z > 0.02` crop，去掉大量桌面低点。
- cube center 只作为训练时 auxiliary target，不作为 rollout 输入。
- action head 使用模型预测的 cube bottleneck 和 phase；rollout 时 phase 也由模型预测。

离线 gate：

| mode | action RMSE | phase acc | cube RMSE |
|---|---:|---:|---:|
| normal | `0.103` | `94.3%` | `0.009m` |
| null | `0.184` | `89.5%` | `0.076m` |
| cross_sample | `0.215` | `87.7%` | `0.075m` |

paired normal-vs-cross action L2 为 `0.215`，明显高于未 crop 版本的 `0.016`，说明 crop + cube auxiliary 让 raw pointcloud 模型真正学到了 object geometry。

闭环，150 steps，30 episodes：

| mode | success | mean reward |
|---|---:|---:|
| normal | `2/30` | `23.09` |
| null | `0/30` | `10.66` |
| cross_demo | `0/30` | `6.46` |

解读：

> 这是 raw pointcloud 输入下的弱 positive rollout signal：normal 已经超过 null/cross，但成功率只有 `2/30`，远低于 object-feature learned-phase 的 `19/30`。它说明下一步方向是对的：必须保留 object-centric bottleneck 和空间监督；但简单 PointNet + 单步 action head 仍然不够，需要更强的 pointcloud encoder、temporal aggregation 或 diffusion/ACT action decoder。

### Scaled raw pointcloud teacher distillation

为了确认 `2/30` 是不是主要受数据量限制，又用同一个 geometry controller 重新采集了 `100` 条成功 teacher rollouts：

```text
100 successful teacher episodes
103 attempts
8388 transitions
input = z>0.02 cropped pointcloud xyz/rgb + task state
model = h256 PointNet, 10k steps, phase + cube auxiliary heads
```

离线 gate 更强通过：

| mode | action RMSE | phase acc | cube RMSE |
|---|---:|---:|---:|
| normal | `0.067` | `98.0%` | `0.0079m` |
| null | `0.210` | `95.0%` | `0.0547m` |
| cross_sample | `0.131` | `97.0%` | `0.0590m` |

paired normal-vs-cross action L2 为 `0.120`，paired normal-vs-null L2 为 `0.295`。

闭环，150 steps，两组 30-episode eval seed：

| eval seed | normal | null | cross_demo |
|---|---:|---:|---:|
| `4100` | `8/30` | `1/30` | `1/30` |
| `4500` | `12/30` | `0/30` | `0/30` |
| aggregate | `20/60` | `1/60` | `1/60` |

mean reward 也保持 normal 优势：seed4100 normal `29.27` vs null `16.85` / cross `16.66`；seed4500 normal `35.57` vs null `16.72` / cross `18.02`。

为了让 baseline 更像训练时对照，又补了两个 matched no-depth baselines：同样 `100` 条 teacher 轨迹、同样 h256/10k 模型、同样两个 eval seeds。

| train input | rollout input | aggregate success | mean reward across seeds |
|---|---|---:|---:|
| normal pointcloud xyz/rgb | normal | `20/60` | `32.42` |
| sampled RGB only, xyz zeroed | rgb_only | `1/60` | `16.69` |
| null points + task state/proprio | null | `3/60` | `14.54` |

解读：

> 这是当前最强的 raw-pointcloud learned action 结果。它的证据边界是 ManiSkill geometry-teacher 数据和 PointNet policy，不是 OpenVLA 端到端；但它已经说明：在更合适的 3D-sensitive benchmark 上，扩大 teacher 数据并使用 primary pointcloud action decoder 后，normal depth/pointcloud 可以在闭环 success 上明显超过 eval-time corrupt controls 和 train-time no-depth baselines。

同样的数据还训练了一个 h=8 action-chunk decoder。离线 gate 通过，paired normal-vs-cross step L2 为 `0.188`；但闭环没有超过单步模型：

| setting | normal | null | cross_demo |
|---|---:|---:|---:|
| success100 chunk, execute 1, seed4100 | `2/30` | `1/30` | `1/30` |

因此这次提升主要来自更多 teacher 数据和更稳的单步 action decoder，而不是简单 action chunk。

为了进一步判断瓶颈在 perception 还是 action decoder，又把同一个 learned cube predictor 接到固定 geometry controller：pointcloud 模型只预测 cube center，控制动作仍由几何控制器生成。这个不是 learned action policy，但它能直接测试 raw pointcloud 几何是否足够用于闭环控制。

闭环，150 steps，30 episodes：

| mode | success | mean reward |
|---|---:|---:|
| learned cube normal | `22/30` | `27.86` |
| learned cube null | `1/30` | `18.50` |
| learned cube cross_demo | `0/30` | `10.28` |

解读：

> 这个结果非常关键：raw cropped pointcloud 的 learned perception 已经足够支撑控制，normal 明显超过 null/cross。30 条 teacher 轨迹时，同类 learned action head 只有 normal `2/30`；扩到 100 条 teacher 轨迹后，learned action head 提升到 aggregate normal `20/60`、null/cross 各 `1/60`。这说明主要瓶颈已经不是 depth perception，而是 learned action / temporal decoder 的数据和结构质量。下一步最该做的是把这个 object-centric bottleneck 接到 ACT/DP3/diffusion action policy，而不是继续证明 depth 有没有几何信号。

## Geometry Controller Positive Diagnostic

最后补了一个 pointcloud geometry controller：用 pointcloud segmentation 估计 cube 位置，并使用 observation 里的 `tcp_pose` / `goal_pos` 做显式几何状态机控制。这不是 learned policy，也不是 VLA 结果；它用于回答一个更基础的问题：

> PickCube 是否真的能被当前 RGB-D/pointcloud 几何闭环解决？

10 episodes 结果：

| mode | success | mean reward |
|---|---:|---:|
| normal pointcloud | `7/10` | `26.38` |
| null pointcloud | `0/10` | `5.84` |
| cross_demo pointcloud | `0/10` | `9.23` |

加入 last-cube memory 和更长 150-step horizon 后：

| mode | success | mean reward |
|---|---:|---:|
| normal pointcloud | `8/10` | `34.70` |
| null pointcloud | `0/10` | `8.76` |
| cross_demo pointcloud | `1/10` | `17.66` |

解读：

> 这是强 positive diagnostic。它说明 ManiSkill3 PickCube 不是“几何没用”或“环境不可解”；真实点云几何足以闭环完成任务，而且 null/cross-demo 明显落后。结合 learned-phase object-feature 的 `19/30`、scaled raw cropped pointcloud action head 的 `20/60`、以及 learned cube + fixed controller 的 `22/30`，失败点已经进一步收紧到“怎样从 raw RGB-D/pointcloud 学稳定 action/temporal decoder”，而不是“depth perception 是否有用”。

## 复现命令

生成 pointcloud demo：

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

训练 gate：

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

## 答辩时怎么说

可以这样说：

> 最终最重要的正结果是：depth/pointcloud 在 ManiSkill3 PickCube 这种 3D-sensitive benchmark 上确实有因果作用。learned-phase object-feature policy 达到 normal `19/30`、null/cross `0/30`；更接近 raw 输入的 cropped pointcloud teacher policy 在扩大到 `100` 条成功轨迹后达到 normal `20/60`，eval-time null/cross 各 `1/60`，matched RGB-only/null train baselines 为 `1/60`/`3/60`。所以结论不是“depth 没用”，而是：depth 需要合适的数据集、object-centric 表示和 primary/temporal action decoder，不能只作为 LIBERO/OpenVLA-OFT 上的可选 residual。

如果需要补边界，再加一句：

> 这个结果的边界是它还不是 OpenVLA 端到端 RGB-D baseline；它已经证明了真实点云几何可以通过 learned policy 转成闭环收益。
