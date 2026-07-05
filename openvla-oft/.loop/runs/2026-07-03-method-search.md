# 2026-07-03 Method Search: RGB-D / 3D VLA Repair

## 目标

回应当前失败点：LIBERO 已经饱和，RLBench reach RGB-only 能闭环成功，但 RGB-D normal 失败、null 成功。搜索更合理的 RGB-D/VLA 方法，决定下一轮最小实验。

## 使用来源

只使用官方项目页、官方仓库、arXiv/论文页：

- PerAct: https://peract.github.io/
- RVT: https://robotic-view-transformer.github.io/
- Act3D: https://act3d.github.io/
- 3D Diffusion Policy: https://3d-diffusion-policy.github.io/
- PointVLA: https://pointvla.github.io/
- SpatialVLA: https://spatialvla.github.io/
- BridgeVLA paper/repo: https://arxiv.org/abs/2506.07961, https://github.com/BridgeVLA/BridgeVLA
- RLBench: https://github.com/stepjam/RLBench
- ManiSkill3: https://github.com/haosulab/ManiSkill
- CALVIN: https://github.com/mees/calvin

## 关键发现

1. 成功的 3D manipulation 方法普遍把 observation 和 action 放进同一个空间结构。
   - PerAct：voxel observation -> next-best voxel action。
   - Act3D：3D feature field -> coarse-to-fine 3D keypose/action detection。
   - BridgeVLA：3D point cloud 投影为多视角 2D 输入，输出 2D heatmap，做 input-output alignment。

2. VLA 加 3D 的方法强调保护原本的 2D/RGB action 能力。
   - PointVLA 冻结 vanilla action expert，用轻量模块注入 point cloud。
   - 这正好解释我们当前现象：RGB-only reach 已经能成功，RGB-D normal 不能大幅扰动这条路径。

3. 单纯换数据集不够。
   - LIBERO 必须降级，因为 RGB-only 接近天花板。
   - 但 RLBench reach 的结果说明，换到非饱和 benchmark 后，depth fusion 仍然会失败。
   - 下一步必须先让 depth 作为 bounded correction，而不是主导 action hidden state。

## 下一轮最小实验

先不扩 stable6，先做 reach-only safe RGB-D repair：

```bash
HDF5_DIR=/root/RLBench/rgbd_hdf5_reach_3demos_64 \
DATASET_NAME=rlbench_reach_3demos_64 \
RUN_ROOT_DIR=/root/runs_rlbench_reach_safe \
MAX_STEPS=5000 \
SAVE_FREQ=1000 \
DEPTH_ACTION_FUSION_GATE_INIT=0.01 \
DEPTH_HIDDEN_DELTA_CLIP=0.05 \
DEPTH_ACTION_RESIDUAL_CLIP=0.02 \
DEPTH_AUX_SPATIAL_LOSS_WEIGHT=0.2 \
DEPTH_DROPOUT=0.2 \
FREEZE_VLA_LORA=True \
FREEZE_PROPRIO_PROJECTOR=True \
FREEZE_ACTION_HEAD_BASE=True \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgbd
```

评估必须包含：

```bash
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh eval-rgbd
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh eval-rgbd-null
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh eval-rgbd-cross-sample
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh diagnose-rgbd-all-strict
```

## Go / No-Go

GO:

- `reach_target` RGB-D normal 达到 `1/1`。
- normal 不低于 null。
- normal 与 cross-sample depth 在 action diagnostic 或 rollout 上出现可解释差异。

NO-GO:

- normal 仍失败、null 成功：depth residual 仍在伤害控制。
- normal/null/cross-sample action 几乎重合：depth 仍未因果进入 action。
- normal 成功但 cross-sample 也成功：不能 claim depth 内容有效。

## 后续分支

如果 safe residual 仍失败：

1. 先实现 residual clamp，而不是扩大训练。
2. 再做 projected multi-view heatmap / 3D action map auxiliary。
3. 如果 RLBench 工程吞吐成为瓶颈，再并行接 ManiSkill3；不要因为 LIBERO 熟悉就回去刷 saturated benchmark。
