# RLBench RGB-D 执行清单

更新时间：2026-07-03

目标：

> 用 RLBench 替代已饱和的 clean LIBERO，扩大训练量，并验证 RGB-D normal 能否超过 matched RGB-only，同时超过 null/shuffle depth。

边界：

> 现在不再把 LIBERO 作为主实验数据集。LIBERO 只保留 sanity check、历史对照和回归测试用途；如果一个方法只能在 LIBERO 上好看，但不能在 RLBench/ManiSkill 这类未饱和 3D benchmark 上通过 causal gate，就不能作为 DepthVLA 的主要结论。

## 1. 当前环境状态

已检查：

```bash
/root/miniconda3/envs/depthvla/bin/python experiments/robot/rlbench/check_rlbench_env.py
```

结果：

- `h5py`: ok
- `numpy`: ok
- `rlbench`: ok
- `pyrep`: ok
- `peract_colab`: ok
- `yarr`: ok
- `pyrep_cffi`: ok
- `get_demo`: ok
- `COPPELIASIM_ROOT`: `/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04`
- `DISPLAY`: `:1.0`
- BridgeVLA RLBench helper: found at `/root/autodl-tmp/BridgeVLA/finetune/RLBench`

结论：

> RLBench/PyRep/peract/yarr Python 环境已经打通。CoppeliaSim 已安装到 `/root`，PyRep CFFI 已构建成功。真实 demonstrations、HDF5 转换、offline probe、matched 小训练和 rollout 入口都已经跑通。下一步瓶颈是 policy-learning / execution-stability，而不是环境安装。

已完成 smoke：

- RLBench headless launch/reset：通过。
- `reach_target` 1 task x 1 demo live collection：通过。
- raw RLBench demo -> DepthVLA HDF5 conversion：通过。
- HDF5 strict validation：通过。
- dense depth -> keypose probe 代码路径：通过。
- stable 6 tasks x 1 demo pilot raw collection：通过。
- stable 6 tasks x 1 demo HDF5 strict validation：通过，`6` files / `6` demos / `632` transitions。
- stable 6 tasks x 1 demo dense keypose probe：通过。
- stable 6 tasks x 3 demos raw collection：通过。
- stable 6 tasks x 3 demos HDF5 strict validation：通过，`6` files / `18` demos / `2009` transitions。
- stable 6 tasks x 3 demos revised dense keypose probe：通过。
- RGB-only `MAX_STEPS=1` training smoke：通过。
- RGB-D dense point + object-query + absolute keypose `MAX_STEPS=1` training smoke：通过。
- stable 6 tasks x 3 demos RGB-only `2000` steps matched training：通过。
- stable 6 tasks x 3 demos RGB-D dense point `2000` steps matched training：通过。
- stable 6 tasks x 3 demos `1 episode/task` rollout gate：`NO-GO`，RGB-only/RGB-D normal/null/shuffle 全部 `0/6`。
- reach-only HDF5 subset：通过，`3 demos / 120 transitions`，路径 `/root/RLBench/rgbd_hdf5_reach_3demos_64`。
- reach-only RGB-only `5000` step overfit：通过。
- reach-only RGB-only closed-loop eval：`MAX_DELTA_XYZ=0.03/0.05/0.08` 均为 `1/1`，成功步数均为 `29`。
- reach-only RGB-D dense/keypose `5000` step overfit：训练完成，但 rollout `NO-GO`。
- reach-only RGB-D closed-loop eval：
  - normal depth：`MAX_DELTA_XYZ=0.03/0.05/0.08` 全部 `0/1`，均跑满 `150` 步无错误。
  - null depth：`MAX_DELTA_XYZ=0.05` 为 `1/1`，`31` 步。
  - shuffle depth：`MAX_DELTA_XYZ=0.05` 为 `0/1`，跑满 `150` 步。

当前 pilot probe 结果：

```text
data_dir: /root/RLBench/rgbd_hdf5_stable6_3demos_64
samples train/test: 1600 / 400
normal xyz_rmse:          0.0803
null xyz_rmse:            0.6097
shuffle_samples xyz_rmse: 0.2035
normal advantage over null:    +0.5294
normal advantage over shuffle: +0.1232
```

说明：

> 对 dense point encoder 来说，单帧内 token/像素 shuffle 会保留点云集合或 depth 直方图，消融太弱；当前 probe 已改用跨样本 shuffle，把 depth/camera/mask 从另一个样本替换进来，更适合作为 causal gate。

结论：

> Dense point depth tokens + absolute keypose auxiliary target 在真实 RLBench RGB-D 数据上已经通过 offline causal gate，并且 RGB/RGB-D 两条训练入口都已通过 1-step smoke。第一组 `6 tasks x 3 demos` matched rollout 没有过关：RGB-only、RGB-D normal、RGB-D null、RGB-D shuffle 都是 `0/6`。

重要修正：

> 早期 `25 max steps` rollout gate 太短，不足以评价 RLBench policy。demo replay 证明真实 next-absolute-pose action 在相同 planning action mode 下需要 `30-136` 步成功，因此 eval 默认 horizon 已改为 `150`。

`max_steps=200` demo replay 结果：

| 任务 | 成功步数 |
|---|---:|
| `slide_block_to_target` | `80` |
| `turn_tap` | `136` |
| `close_jar` | `120` |
| `open_drawer` | `89` |
| `reach_target` | `30` |
| `pick_up_cup` | `91` |

`max_steps=150` matched rollout 结果：

| policy / depth mode | success | 主要错误 |
|---|---:|---|
| RGB-only | `0/6` | `InvalidActionError` |
| RGB-D normal | `0/6` | `InvalidActionError` |

Reach-only trace：

| policy / depth mode | success | length | delta xyz mean | gripper mean |
|---|---:|---:|---:|---:|
| RGB-only | `0/1` | `59` | `0.01435` | `1.0000` |
| RGB-D normal | `0/1` | `70` | `0.01109` | `0.8566` |
| RGB-D null | `0/1` | `72` | `0.01135` | `0.8956` |

Offline policy-vs-demo action diagnostic 已新增：

```text
experiments/robot/rlbench/diagnose_policy_actions.py
```

当前 smoke 结果：

| policy / depth mode | samples | 覆盖 | xyz RMSE | xyz direction cosine | gripper abs error |
|---|---:|---|---:|---:|---:|
| RGB-only | `12` | `6 tasks x 2 samples` | `0.00564` | `0.9259` | `0.1738` |
| RGB-D normal | `12` | `6 tasks x 2 samples` | `0.00514` | `0.6051` | `0.1331` |
| RGB-D null | `12` | `6 tasks x 2 samples` | `0.00515` | `0.6150` | `0.1201` |
| RGB-D shuffle | `12` | `6 tasks x 2 samples` | `0.00514` | `0.6051` | `0.1331` |

这还不是最终统计结论，但已经有两个清晰信号：

1. 单步 delta 量级并不爆炸，xyz RMSE 在毫米级，当前 rollout 更像方向误差和闭环漂移累积。
2. RGB-D normal/null/shuffle 几乎重合，说明当前 RGB-D checkpoint 的 action prediction 仍没有因果依赖 depth 内容。

当前解释：

> 这不是 depth probe 的失败，也不是 action mode 完全不可执行。更准确地说：offline keypose probe 证明 depth 有空间信号，但当前 VLA action head 没有把这个信号用到 action prediction；同时 `3 demos/task + 2000 steps` 的 policy 在 closed-loop 中会逐步漂移，最后进入 planner infeasible state。下一步应该先让 `reach_target` 单任务学到成功，并加强 depth-action coupling，再扩大到 stable6。

更新：

> `reach_target` RGB-only 单任务 overfit 已经通过，说明 RLBench action adapter 本身可以闭环成功。但 reach-only RGB-D dense/keypose 目前没有通过：normal depth 三个 clip 全失败，null depth 反而成功。当前下一步不是扩大数据，而是修 RGB-D fusion/head，让 depth 不再破坏成功的 RGB anchor。

当前 eval 文件：

```text
/root/autodl-tmp/openvla-oft/experiments/logs/rlbench_eval_results/rgb_only.json
/root/autodl-tmp/openvla-oft/experiments/logs/rlbench_eval_results/rgbd_normal.json
/root/autodl-tmp/openvla-oft/experiments/logs/rlbench_eval_results/rgbd_null.json
/root/autodl-tmp/openvla-oft/experiments/logs/rlbench_eval_results/rgbd_shuffle.json
/root/autodl-tmp/openvla-oft/experiments/logs/rlbench_eval_results/rgbd_causal_gate.json
/root/autodl-tmp/openvla-oft/experiments/logs/rlbench_eval_results_h150/rgb_only.json
/root/autodl-tmp/openvla-oft/experiments/logs/rlbench_eval_results_h150/rgbd_normal.json
/root/autodl-tmp/openvla-oft/experiments/logs/rlbench_eval_reach_h150_trace/rgb_only.json
/root/autodl-tmp/openvla-oft/experiments/logs/rlbench_eval_reach_h150_trace/rgbd_normal.json
/root/autodl-tmp/openvla-oft/experiments/logs/rlbench_eval_reach_h150_trace/rgbd_null.json
```

## 2. 需要先安装/配置

参考脚本：

```text
/root/autodl-tmp/BridgeVLA/finetune/RLBench/install_rlbench.sh
```

需要的核心组件：

1. CoppeliaSim Edu `V4_1_0`
2. PyRep
3. RLBench
4. YARR
5. peract_colab helper
6. headless display / xvfb
7. `COPPELIASIM_ROOT`
8. `LD_LIBRARY_PATH`
9. `QT_QPA_PLATFORM_PLUGIN_PATH`
10. `DISPLAY`

安装后再次检查：

```bash
/root/miniconda3/envs/depthvla/bin/python experiments/robot/rlbench/check_rlbench_env.py
```

必须看到：

```text
rlbench      ok
pyrep        ok
peract_colab ok
yarr         ok
```

当前仓库提供了更安全的安装/检查入口：

```bash
experiments/robot/rlbench/setup_rlbench_env.sh
```

如果要把 CoppeliaSim 放到空间更宽的 `/root`，建议：

```bash
INSTALL=0 DOWNLOAD_COPPELIASIM=1 \
COPPELIASIM_DIR=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 \
COPPELIASIM_ARCHIVE=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
experiments/robot/rlbench/setup_rlbench_env.sh
```

CoppeliaSim 下载并解压后，再构建 PyRep CFFI：

```bash
BUILD_PYREP=1 \
COPPELIASIM_DIR=/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 \
experiments/robot/rlbench/setup_rlbench_env.sh
```

注意：

> 不要直接运行 BridgeVLA 原始 `pip install -e ...` 全套安装命令。它可能升级 numpy/torch，破坏当前 DepthVLA 环境。本仓库脚本默认只走 `PYTHONPATH`，需要 editable install 时也使用 `--no-deps`。

如果 headless display 没起来：

```bash
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh xvfb
```

## 3. 第一批 RLBench 任务

先不要直接上 MT100。第一轮只做 6 个已通过 live-demo smoke 的稳定 pilot 任务：

```text
slide_block_to_target
turn_tap
close_jar
open_drawer
reach_target
pick_up_cup
```

每个任务：

- train: `10 demos/task`
- eval: `25 episodes/task`
- 总 eval: `150 episodes`

这比 LIBERO clean 更能暴露 depth 的价值，因为任务依赖：

- 平面目标位置与深度关系
- 旋钮/抽屉/罐子等 articulation 几何
- 目标点 reaching 的 metric 3D sanity check
- 杯子抓取中的接触和相对位姿

## 4. 数据转换

新增转换脚本：

```text
experiments/robot/rlbench/convert_rlbench_to_hdf5.py
```

推荐优先使用 stage runner：

```text
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh
experiments/robot/rlbench/rlbench_stage.env.example
```

先 dry-run 检查命令：

```bash
DRY_RUN=1 experiments/robot/rlbench/run_rlbench_rgbd_stage.sh dry-run
```

如果本地还没有 RLBench demos，先生成 pilot 数据：

```bash
DATA_ROOT=/root/autodl-tmp/RLBench/peract_dataset/all_variations_128 \
TASKS=slide_block_to_target,turn_tap,close_jar,open_drawer,reach_target,pick_up_cup \
MAX_DEMOS_PER_TASK=10 \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh generate-demos
```

真实 smoke 已验证过官方 generator 输出可以被当前 converter 读取。

为了避免 RLBench 官方 generator 在 multi-task 模式下卡住，当前 `generate-demos` 已改为逐任务生成。这样某个任务失败时更容易定位，也方便断点续跑。

示例命令：

```bash
/root/miniconda3/envs/depthvla/bin/python experiments/robot/rlbench/convert_rlbench_to_hdf5.py \
  --data_root /path/to/RLBench/peract_dataset/all_variations_128 \
  --target_dir /root/autodl-tmp/RLBench/rgbd_hdf5_6tasks_10demos \
  --split train \
  --tasks slide_block_to_target,turn_tap,close_jar,open_drawer,reach_target,pick_up_cup \
  --cameras front,wrist \
  --max_demos_per_task 10 \
  --overwrite
```

输出字段兼容当前 DepthVLA HDF5 风格：

- `data/demo_x/actions`
- `data/demo_x/obs/agentview_rgb`
- `data/demo_x/obs/eye_in_hand_rgb`
- `data/demo_x/obs/agentview_depth_m`
- `data/demo_x/obs/eye_in_hand_depth_m`
- `data/demo_x/obs/agentview_K`
- `data/demo_x/obs/eye_in_hand_K`
- `data/demo_x/obs/agentview_T_camera_to_base`
- `data/demo_x/obs/eye_in_hand_T_camera_to_base`
- `data/demo_x/obs/proprio`

额外保存给新 action-space 实验：

- `data/demo_x/obs/rlbench_abs_gripper_pose`
- `data/demo_x/obs/rlbench_next_abs_gripper_pose`
- `data/demo_x/rlbench_delta_action`
- `data/demo_x/rlbench_keypose_action`

## 5. 数据验证

转换后先不要训练，先写/跑 validator，检查：

1. HDF5 文件数量 = 任务数量。
2. 每个任务 demo 数量 = `10`。
3. RGB shape 是 `H x W x 3`。
4. Depth shape 是 `H x W`，且非零、有限。
5. K shape 是 `3 x 3`。
6. T shape 是 `4 x 4`。
7. action shape 是 `T x 7`。
8. keypose action shape 是 `T x 8`。
9. delta action 数值范围合理，不爆炸。
10. absolute gripper pose 在 RLBench scene bounds 内。

已新增 validator：

```text
experiments/robot/rlbench/validate_rlbench_hdf5.py
```

示例命令：

```bash
/root/miniconda3/envs/depthvla/bin/python experiments/robot/rlbench/validate_rlbench_hdf5.py \
  --data_dir /root/autodl-tmp/RLBench/rgbd_hdf5_6tasks_10demos \
  --strict
```

已新增 dataset smoke：

```text
experiments/robot/rlbench/smoke_rlbench_hdf5_dataset.py
```

它验证三件关键兼容性：

1. RLBench gripper 不按 LIBERO 规则反转。
2. RLBench RGB 不做 LIBERO 的 180 度旋转。
3. `absolute_keypose` auxiliary label 能正确读到 8 维 keypose。

运行：

```bash
/root/miniconda3/envs/depthvla/bin/python experiments/robot/rlbench/smoke_rlbench_hdf5_dataset.py
```

已新增 dense depth -> absolute keypose offline probe：

```text
experiments/robot/rlbench/probe_dense_depth_keypose.py
```

真实 RLBench HDF5 转换完成后，先运行：

```bash
/root/miniconda3/envs/depthvla/bin/python experiments/robot/rlbench/probe_dense_depth_keypose.py \
  --data_dir /root/autodl-tmp/RLBench/rgbd_hdf5_6tasks_10demos \
  --max_samples 2000 \
  --num_points_per_view 512 \
  --token_dim 128 \
  --hidden_dim 256 \
  --batch_size 32 \
  --epochs 20 \
  --device cuda \
  --threshold 0.01 \
  --output /root/autodl-tmp/openvla-oft/experiments/logs/rlbench_dense_keypose_probe.json
```

GO 条件：

```text
normal xyz_rmse 至少比 null 低 0.01m
normal xyz_rmse 至少比 shuffle 低 0.01m
```

如果这个 gate 不过，不要直接跑 VLA 大训练。

最终 rollout gate 已新增：

```text
experiments/robot/rlbench/compare_rgbd_rollout_results.py
```

stage runner 入口：

```bash
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh gate-results
```

默认读取：

```text
experiments/logs/rlbench_eval_results/rgb_only.json
experiments/logs/rlbench_eval_results/rgbd_normal.json
experiments/logs/rlbench_eval_results/rgbd_null.json
experiments/logs/rlbench_eval_results/rgbd_shuffle.json
```

通过条件：

- `rgbd_normal - rgb_only >= 0.05`
- `rgbd_normal - rgbd_null >= 0.05`
- `rgbd_normal - rgbd_shuffle >= 0.05`

## 6. 模型路线

第一轮必须做 matched comparison：

### A. RGB-only baseline

输入：

- RGB front + wrist
- language
- proprio

输出：

- delta action chunk

目的：

> 得到新的非饱和 baseline。

### B. RGB-D dense tokens

输入：

- RGB front + wrist
- depth front + wrist
- K/T camera geometry
- language
- proprio

表示：

- depth -> base-frame point tokens
- 不做 `4x4` grid-pooled summary
- 使用 1024-4096 sampled dense 3D tokens

输出：

- delta action chunk

目的：

> 验证 dense 3D 表示是否比旧 grid pooling 更可用。

已新增模块：

```text
prismatic/models/dense_point_depth_encoder.py
```

训练入口已支持：

```text
--depth_encoder_type dense_point
--depth_num_points_per_view 1024
```

最小 smoke 已通过：

```text
DensePointDepthTokenEncoder -> object_query action head -> delta actions + absolute keypose aux
tokens:  (B, V*N, D)
actions: (B, 8, 7)
keypose: (B, 8)
```

### C. RGB-D hybrid action

输入同 B。

输出：

- 主输出：delta action chunk
- 辅助输出：absolute keypose / heatmap / 3D action map

目的：

> 让 depth 必须解决 3D grounding 子问题，而不是只做可选扰动。

训练时使用：

```text
--aux_target absolute_keypose
--aux_output_dim 8
--depth_aux_spatial_loss_weight 0.05
```

对应 HDF5 字段：

```text
data/demo_x/rlbench_keypose_action
```

## 7. 训练规模

不要从超大规模开始，按 gate 扩大：

| 阶段 | 数据 | 训练目标 |
|---|---|---|
| Smoke | `1 task x 2 demos` | 数据和 forward 跑通 |
| Pilot | `6 tasks x 10 demos` | normal 是否超过 RGB/null/shuffle |
| Main-10 | `18 tasks x 10 demos` | 和 PerAct/RVT 常用设置对齐 |
| Main-100 | `18 tasks x 100 demos` | 扩大训练量，争取稳定超过 RGB |

## 8. 成功标准

Pilot 阶段硬门槛：

```text
RGB-D normal >= RGB-only + 5 percentage points
RGB-D normal >= RGB-D null + 5 percentage points
RGB-D normal >= RGB-D shuffle + 5 percentage points
```

如果用 `150 eval episodes`：

```text
normal 至少比 RGB-only 多成功 8 episodes
normal 至少比 null 多成功 8 episodes
normal 至少比 shuffle 多成功 8 episodes
```

如果不满足：

- normal ≈ null：depth 仍被忽略。
- shuffle >= normal：depth 分支学到非因果 artifact。
- normal > null/shuffle 但 < RGB-only：depth 有信号，但融合破坏 RGB policy。

## 9. Action Space 对照

必须比较：

1. `delta_7`
   - `[dx, dy, dz, droll, dpitch, dyaw, gripper]`
   - 和当前 OpenVLA-OFT 最兼容。

2. `absolute_keypose_8`
   - `[x, y, z, qx, qy, qz, qw, gripper]`
   - 和 metric depth / RLBench keypose 最直接对齐。

3. `hybrid`
   - 执行仍用 delta action。
   - 训练增加 absolute keypose 或 heatmap 辅助头。
   - 最推荐。

不要直接押注纯 absolute，因为 absolute action 学习更难；也不要只做 delta，因为 delta 太容易绕开 depth。

## 10. 下一步代码任务

现在不要回头刷 LIBERO，也不要直接上 `18 tasks x 100 demos`。按下面顺序把 RLBench 闭环 policy 先救起来：

1. 使用 reach-only 子集做最小闭环 overfit。
   - 子集已创建：

     ```bash
     SUBSET_HDF5_SOURCE=/root/RLBench/rgbd_hdf5_stable6_3demos_64 \
     HDF5_DIR=/root/RLBench/rgbd_hdf5_reach_3demos_64 \
     SUBSET_TASKS=reach_target \
     experiments/robot/rlbench/run_rlbench_rgbd_stage.sh make-subset
     ```

   - 验证已通过：

     ```text
     /root/RLBench/rgbd_hdf5_reach_3demos_64
     1 file / 3 demos / 120 transitions
     ```

2. 跑 policy-vs-demo offline action diagnostic。
   - 在 `/root/RLBench/rgbd_hdf5_stable6_3demos_64` 上加载 checkpoint。
   - 对比 predicted action chunk 第一步和 demo delta。
   - 报告 xyz/rpy/gripper 的 MAE/RMSE、方向 cosine、norm 分布。

3. 训练 `reach_target` 单任务。
   - 先用 `3 demos` 确认能不能 overfit 到 closed-loop success。
   - 不成功就不要扩大 stable6。
   - RGB-only overfit 命令模板：

     ```bash
     HDF5_DIR=/root/RLBench/rgbd_hdf5_reach_3demos_64 \
     DATASET_NAME=rlbench_reach_3demos_64 \
     RUN_ROOT_DIR=/root/runs_rlbench_reach_3demos \
     MAX_STEPS=5000 \
     SAVE_FREQ=1000 \
     experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgb
     ```

   - RGB-D overfit 命令模板：

     ```bash
     HDF5_DIR=/root/RLBench/rgbd_hdf5_reach_3demos_64 \
     DATASET_NAME=rlbench_reach_3demos_64 \
     RUN_ROOT_DIR=/root/runs_rlbench_reach_3demos \
     MAX_STEPS=5000 \
     SAVE_FREQ=1000 \
     experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgbd
     ```

4. 跑 action execution sensitivity。
   - 对同一个 checkpoint 分别评估：

     ```bash
     MAX_DELTA_XYZ=0.03
     MAX_DELTA_XYZ=0.05
     MAX_DELTA_XYZ=0.08
     ```

   - 目标是减少长时序漂移后触发 `InvalidActionError`。

5. 如果 reach 单任务能成功，再生成 `6 tasks x 10 demos`。
   - matched RGB-only。
   - matched RGB-D dense point + absolute keypose aux。
   - normal/null/shuffle rollout。

6. 只有 stable6 通过 gate，再扩到 PerAct/RVT 常用 `18 tasks x 10 demos`。

核心原则：

> LIBERO 已经不是主战场；RLBench 当前也不是 depth claim 阶段，而是先让 OpenVLA-OFT action policy 在未饱和的 3D benchmark 上闭环可执行。换数据集之后仍然必须保留 matched RGB-only、normal/null/shuffle 和 closed-loop rollout 三个硬门槛。

当前代码状态：

- RLBench/PyRep/CoppeliaSim 环境已打通。
- 真实 `stable6 x 3 demos` 数据已生成并转换。
- Dense point depth offline causal gate 已通过。
- RGB-only/RGB-D matched 小训练已完成。
- Eval runner 默认 horizon 已改为 `150`，并暴露 `MAX_DELTA_XYZ`。
- Offline action prediction diagnostic 已新增；RGB-only/RGB-D normal/null/shuffle balanced 12-sample smoke 已通过。
- Stage runner 已暴露 `DEPTH_AUX_SPATIAL_LOSS_WEIGHT`，当前 reach-only RGB-D overfit 使用 `0.2`，用于加强 absolute-keypose depth grounding。
- Stage runner 已新增 `cross_sample` depth corruption：
  - `diagnose-rgbd-cross-sample`
  - `eval-rgbd-cross-sample`
  - `diagnose-rgbd-all-strict`
  - `eval-rgbd-all-strict`
- Stage runner 已新增 safe RGB-D 训练控制：
  - `DEPTH_ACTION_FUSION_GATE_INIT`
  - `DEPTH_DROPOUT`
  - `FREEZE_VLA_LORA`
  - `FREEZE_PROPRIO_PROJECTOR`
  - `FREEZE_ACTION_HEAD_BASE`
- 下一步缺口是运行 safe reach-only RGB-D repair；如果 normal depth 不能先在 reach-only 上不输 null，就不要扩大 stable6，更不要回头刷 LIBERO。
