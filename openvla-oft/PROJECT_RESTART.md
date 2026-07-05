# DepthVLA-OFT 重新整理起点

更新时间：2026-06-16

## 1. 文档状态

旧的实验总结、汇报提纲、Q&A、8 小时 sprint 计划等 Markdown 文件已经统一归档到：

```text
docs_archive_20260616_restart/
```

根目录现在只保留原仓库基础文档，以及这个新的重新整理入口。

## 2. 当前保留结论

先保留三个经过实验支持的判断：

1. RGB-only OpenVLA-OFT 在 clean LIBERO 上已经很强，clean tasks 接近刷满。
2. Depth 数据本身不是无效的，offline probe 显示 normal depth 比 null/shuffle 更有几何和 action 预测信号。
3. 当前轻量 depth fusion 没有在 rollout 中形成稳定因果收益，normal depth 经常不优于 null/shuffle。

一句话：

> clean LIBERO 更像 pipeline sanity check，不适合作为证明 depth 边际价值的最终 benchmark。

当前决策：

> 后续主线不再使用 LIBERO 作为核心数据集。所有主要工程和实验判断优先围绕 RLBench；ManiSkill3 作为吞吐和规模扩展候选，RoboCasa/CALVIN 作为后续泛化或 action-space 对照。

## 3. 重新开始的方向

下一轮不要继续堆同一套旧文档和旧实验叙事，先重新定义问题：

> 在 RGB-only 没有饱和、且强依赖 3D 几何的评测中，metric depth 是否能带来可验证的 causal gain？

优先考虑：

1. 换更 3D-sensitive 的评测或数据集。
2. 保留 normal/null/shuffle 消融作为硬门槛。
3. 使用更强空间表示，而不是只依赖 grid-pooled depth summary。
4. 先小规模验证，再决定是否扩大训练。

## 4. 待定问题

重新开始前需要明确：

1. 下一轮主 benchmark 选什么？
2. 是否继续基于 OpenVLA-OFT，还是只保留经验教训换实现路线？
3. depth 表示用 dense feature、point rendering、Ego3D encoding，还是 heatmap-style bottleneck？
4. 最小可行实验的成功标准是什么？

## 5. 建议下一步

先写一个新的实验设计文档，只回答三件事：

1. 选哪个数据集/评测。
2. 为什么这个评测能体现 depth 价值。
3. 第一轮最小实验怎么做，如何用 normal/null/shuffle 判断是否继续。

当前新的实验路线文档：

```text
NEXT_RGBD_BENCHMARK_PLAN.md
```
