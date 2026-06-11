# Related Work Comparison: 3D-Enhanced Vision-Language-Action Models

## Overview
This document compares our DepthVLA approach with related work in 3D-enhanced robot manipulation, particularly focusing on how different methods integrate depth/3D information into vision-language-action models.

---

## 1. Base VLA Models

### 1.1 OpenVLA [1]
**Paper**: "OpenVLA: An Open-Source Vision-Language-Action Model" (arXiv:2406.09246)
- **Official page**: https://openvla.github.io/
- **Model**: https://huggingface.co/openvla/openvla-7b

**Architecture**:
- 7B-parameter model combining Llama 2 LLM with visual encoders (DINOv2 + SigLIP)
- Trained on 970k real-world robot demonstrations (Open X-Embodiment dataset)
- Inputs: language instructions + camera images → robot actions

**Performance**:
- Outperforms RT-2-X (55B) by 16.5% despite 7x fewer parameters
- Strong generalist manipulation across 29 tasks and multiple embodiments

**Relation to our work**:
- DepthVLA builds directly on OpenVLA architecture
- We keep the RGB-language processing unchanged and add depth as auxiliary modality
- Our base RGB-only checkpoint uses OpenVLA-OFT training recipe

### 1.2 OpenVLA-OFT [2]
**Paper**: "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success" (arXiv:2502.19645)
- **Code**: https://github.com/moojink/openvla-oft
- **Project**: https://openvla-oft.github.io/

**Key Optimizations**:
- **Parallel decoding** for faster inference
- **Action chunking** (predicting multiple actions at once)
- **Continuous action representation** (L1 regression instead of discrete tokens)
- **25-50x faster inference** with 20%+ success rate improvement

**Relation to our work**:
- Our DepthVLA training pipeline is based on OpenVLA-OFT
- We adopt the same L1 regression action head and continuous action representation
- All our baselines use the OFT recipe (action chunking, continuous actions, LoRA fine-tuning)

---

## 2. 3D Representation Methods

### 2.1 Perceiver-Actor (PerAct) [3]
**Paper**: "Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation" (CoRL 2023)
- **Project**: https://peract.github.io/
- **Code**: https://github.com/peract/peract

**Approach**:
- Encodes language goals + **RGB-D voxel observations** using Perceiver Transformer
- Outputs discretized actions via "next best voxel action" detection
- Directly learns perceptual representations without separate detectors/segmentors

**Key differences from DepthVLA**:
- PerAct uses **voxelized 3D space** as native representation
- Trained from scratch on task-specific data
- Not building on pretrained VLM
- Our approach: leverage pretrained VLM, add depth as lightweight auxiliary signal

### 2.2 RVT (Robotic View Transformer) [4]
**Paper**: "RVT: Robotic View Transformer for 3D Object Manipulation" (CoRL 2023)
- **Project**: https://robotic-view-transformer.github.io/
- **Code**: https://github.com/nvlabs/rvt

**Approach**:
- Multi-view transformer with attention across camera views
- **Re-renders input from virtual views** around robot workspace
- Inputs: multi-view images + language → gripper pose actions
- 26% higher success than PerAct on RLBench (18 tasks, 249 variants)

**Key differences from DepthVLA**:
- RVT synthesizes novel views via rendering, not explicit depth encoding
- Attention mechanism aggregates multi-view information
- Not leveraging pretrained VLM knowledge
- Our approach: explicit metric depth + camera geometry, preserves VLM prefix

### 2.3 3D Diffusion Policy (DP3) [5]
**Paper**: "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations" (arXiv:2403.03954)

**Approach**:
- Combines **3D visual representations** with **diffusion-based action generation**
- Maintains practical inference speeds
- Effective across simulation and real-world with fewer demonstrations

**Follow-up work**:
- iDP3: Improved 3D Diffusion Policy for humanoid manipulation
- Extensions for coverage path planning and frequency-aware variants

**Key differences from DepthVLA**:
- Uses diffusion models for action generation (vs. our transformer-based regression)
- Does not leverage pretrained VLM
- 3D representations are integral to the model (not auxiliary)
- Our approach: keep VLM intact, add depth as action-side conditioning

---

## 3. Recent 3D-Enhanced VLA Models

### 3.1 PointVLA [6]
**Paper**: "PointVLA: Injecting the 3D World into Vision-Language-Action Models" (IEEE RA-L 2026)
- **arXiv**: https://arxiv.org/abs/2503.07511

**Approach**:
- Enhances pretrained VLA with **3D point cloud inputs**
- **No retraining**: lightweight modular block with frozen action expert
- **Selective skip-block fusion** to inject 3D embeddings into transformer blocks

**Performance**:
- Strong results in long-horizon tasks (picking/packing from moving conveyor belts)
- Demonstrates generalization across complex, dynamic environments

**Key differences from DepthVLA**:
- Uses **point clouds** as 3D representation (vs. our metric depth + geometry features)
- Injects 3D via skip connections into transformer blocks
- We inject depth at action head level, not into VLM prefix
- Similar philosophy: augment pretrained VLA without full retraining

**Similarity to our work**:
- Both avoid retraining base VLM
- Both use modular 3D injection
- Both focus on preserving pretrained knowledge while adding spatial reasoning

### 3.2 SpatialVLA [7]
**Paper**: "SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model" (arXiv:2501.15830)
- **Hugging Face**: https://huggingface.co/papers/2501.15830

**Approach**:
- **Ego3D Position Encoding**: novel encoding for spatial information
- **Adaptive Action Grids**: develops generalizable spatial action knowledge
- **Zero-shot performance**: directly applicable without additional training

**Performance**:
- State-of-the-art across diverse evaluations
- Superior zero-shot performance
- **Faster inference with fewer tokens per action**
- Strong multi-task generalization in simulation and real-world

**Key differences from DepthVLA**:
- Ego3D encoding is built into the model architecture
- Focus on spatial position encoding rather than explicit depth
- Emphasizes zero-shot transfer
- Our approach: explicit metric depth supervision with auxiliary losses

**Similarity to our work**:
- Both emphasize spatial understanding for manipulation
- Both aim for generalization across tasks
- Both address limitations of purely 2D vision inputs

### 3.3 BridgeVLA [8]
**Paper**: "BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models" (NeurIPS 2025)
- **arXiv**: https://arxiv.org/abs/2506.07961
- **Code**: https://github.com/BridgeVLA/BridgeVLA
- **Project**: https://bridgevla.github.io/

**Approach**:
- **Input alignment**: projects 3D inputs to multiple 2D images (aligns with VLM capabilities)
- **Output alignment**: uses **2D heatmaps** for action prediction
- Unifies input-output spaces in consistent 2D image format

**Performance**:
- Improves RLBench success rate from 81.4% to 88.2%
- Outperforms baselines across three simulation benchmarks

**Key differences from DepthVLA**:
- BridgeVLA converts 3D → 2D to leverage VLM's image understanding
- Uses heatmap-based action representation
- Our approach: preserve 3D metric information, inject at action head
- We use continuous action regression, not heatmaps

**Similarity to our work**:
- Both aim to efficiently integrate 3D information into VLMs
- Both avoid extensive VLM retraining
- Both focus on preserving pretrained VLM capabilities

---

## 4. Datasets and Benchmarks Comparison

### 4.1 Training Dataset Comparison

| Method | Training Data | Scale | Source | Real/Sim |
|--------|---------------|-------|--------|----------|
| **OpenVLA** | Open X-Embodiment | 970k demos | Multi-robot datasets | Real-world |
| **OpenVLA-OFT** | Task-specific fine-tuning | Varies | LIBERO, custom | Sim + Real |
| **PerAct** | RLBench | 100 demos/task | RLBench simulator | Simulation |
| **RVT** | RLBench | 100 demos/task | RLBench simulator | Simulation |
| **DP3** | Multi-source | 10+ demos/task | 72 sim + 4 real tasks | Sim + Real |
| **PointVLA** | Pretrained VLA + task data | Varies | LIBERO, RLBench | Sim + Real |
| **SpatialVLA** | Custom spatial dataset | Unknown | LIBERO, Simpler | Sim + Real |
| **BridgeVLA** | RLBench + Bridge V2 | 100 demos/task | RLBench, Bridge | Sim + Real |
| **DepthVLA (Ours)** | LIBERO RGB-D | 5 tasks × 20 demos | LIBERO-Spatial | Simulation |

**Key observations**:
- **OpenVLA pretraining**: 970k real-world demos is the largest scale, enables strong generalist policy
- **RLBench standard**: Most 3D methods use RLBench (100 demos/task) for simulation evaluation
- **DP3 efficiency**: Achieves good results with only 10 demos/task
- **Our limitation**: 5 tasks × 20 demos (89 successful demos, 11k transitions) is much smaller scale
  - No pretraining on large-scale data
  - Limited task diversity for LIBERO-Plus generalization

### 4.2 Evaluation Benchmark Comparison

| Method | Primary Benchmark | Tasks Evaluated | Success Metric | Notes |
|--------|------------------|-----------------|----------------|-------|
| **OpenVLA** | Open X-Embodiment | 29 tasks, multi-robot | Task success rate | Real-world evaluation |
| **PerAct** | RLBench | 18 tasks | Task success rate | 249 task variants |
| **RVT** | RLBench | 18 tasks | Task success rate | 26% better than PerAct |
| **RVT-2** | RLBench | Multiple tasks | 82% success rate | Improved from 65% |
| **DP3** | Custom suite | 72 sim + 4 real | 24.2% relative gain | Multiple benchmarks |
| **PointVLA** | LIBERO, custom | Long-horizon tasks | 92.5% success | Conveyor belt tasks |
| **SpatialVLA** | LIBERO, Simpler | Multi-task | SoTA performance | Zero-shot capable |
| **BridgeVLA** | RLBench | Multiple tasks | 88.2% (↑ from 81.4%) | NeurIPS 2025 |
| **DepthVLA (Ours)** | LIBERO-Spatial, LIBERO-Plus | 5 clean + 30 robustness | 100% clean, 70% robust | Focus on robustness |

**Evaluation protocol differences**:
- **RLBench**: Most common simulation benchmark, 18 tasks with variants
- **LIBERO**: Tests task composition and knowledge transfer
- **LIBERO-Plus**: **Our primary contribution** - robustness under 7 perturbation dimensions
- **Real-world**: OpenVLA, PointVLA, DP3 include real robot validation

### 4.3 LIBERO [9]
**Paper**: "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning" (NeurIPS 2023)

**Characteristics**:
- Benchmark for lifelong robot learning
- Tests knowledge transfer across tasks
- Multiple task suites (spatial, object, goal, 10, 90, 100)

**Our usage**:
- Stage 1-2 training: 5 tasks from LIBERO-Spatial with 20 demos each (89 successful)
- Initial evaluation on trained tasks showed 100% success for RGB-only
- **Total training data**: 11,037 transitions across 5 tasks

### 4.4 LIBERO-Plus [10]
**Paper**: "LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models" (arXiv:2510.13626)
- **Code**: https://github.com/sylvestf/LIBERO-plus
- **Dataset**: https://huggingface.co/datasets/Sylvest/LIBERO-plus

**Key Findings**:
- Systematically stress-tests VLA models via **7 perturbation dimensions**
- Exposes extreme sensitivity: 95% → <30% under modest perturbations
- Critical weaknesses: **camera viewpoint**, **robot initial state**, **language instruction** sensitivity

**Perturbation dimensions**:
1. Objects Layout
2. Camera Viewpoints
3. Robot Initial States
4. Lighting conditions
5. Textures
6. Distractors
7. Background variations

**Our usage**:
- Primary evaluation benchmark for robustness testing
- Fixed 30-task probe (10 from each: Objects Layout, Camera Viewpoints, Robot Initial States)
- Revealed RGB-only: 17/30 (56.7%) - much lower than clean tasks
- Our best result: 21/30 (70%) with interpolated RGB-only checkpoint

**Key insight from our experiments**:
- LIBERO-Plus camera/initstate perturbations are where depth should help most
- However, our depth variants struggled: action-summary best depth was 14/30 (46.7%)
- Normal depth ≈ null depth in most experiments, suggesting depth path not yet causally used

---

## 5. Our DepthVLA Approach

### 5.1 Architecture Comparison

| Method | Base Model | 3D Representation | Integration Point | Retraining VLM? |
|--------|-----------|-------------------|-------------------|-----------------|
| **PerAct** | - | RGB-D voxels | Native | N/A (trained from scratch) |
| **RVT** | - | Multi-view rendering | Native | N/A (trained from scratch) |
| **DP3** | - | 3D point features | Native | N/A (diffusion policy) |
| **PointVLA** | Pretrained VLA | Point clouds | Skip-block fusion | No |
| **SpatialVLA** | VLA | Ego3D encoding | Built-in architecture | Yes (trained with Ego3D) |
| **BridgeVLA** | VLM | 3D→2D projection | 2D heatmaps | Limited |
| **DepthVLA (Ours)** | OpenVLA-7B | Metric depth + K/T | Action head | No |

### 5.2 Key Design Choices

**1. Preserve VLM Prefix**:
- Unlike PerAct/RVT/DP3, we keep pretrained VLM knowledge intact
- RGB-language processing unchanged
- Similar philosophy to PointVLA, but different injection point

**2. Action-Side Depth Integration**:
- Explored three fusion modes:
  - `depth_prefix_append`: append depth tokens to VLM prefix (failed: 10/15 clean tasks vs 15/15 RGB)
  - `depth_action_summary`: global depth summary → action residual (best depth: 14/30 on LIBERO-Plus)
  - `depth_object_query`: object-conditioned depth queries (11/30 on LIBERO-Plus)
- Best approach: action-side fusion avoids disrupting pretrained representations

**3. Metric Depth Representation**:
- Extract metric depth from MuJoCo simulator
- Compute base-frame 3D geometry using camera intrinsics K and extrinsics T
- Coarse 4x4 grid pooling per camera view
- Features: [X_base, Y_base, Z_base, z_camera, valid, u_norm, v_norm, view_id]

**4. Auxiliary Supervision**:
- Tested multiple targets:
  - `distance_bin`: gripper-to-contact distance (classification)
  - `contact_xyz`: vector to nearest visible geometry
  - `gripper_to_contact_distance`: scalar distance (regression)
  - `ee_to_object_xyz`, `object_to_target_xyz`: task-relevant 3D vectors
- Best: continuous regression targets over classification bins

### 5.3 Experimental Findings

**Stage 2 Results (LIBERO-Spatial clean tasks 0,1,2,7,9, 3 trials each)**:
- RGB-only: **15/15 (100%)**
- Depth prefix append: 10/15 (66.7%) ❌
- Depth + dataset normalization: 6/15 (40%) ❌
- Action-summary (frozen RGB, 5k): **13/15 (86.7%)** ✓
- Action-summary (full train, 20k): varied by aux target

**LIBERO-Plus Robustness (30-task probe, 1 trial each)**:
- RGB-only (Stage 2): 17/30 (56.7%)
- Action-summary distance-bin normal: 14/30 (46.7%)
- Action-summary null depth: 14/30 (46.7%) ⚠️ *normal ≈ null*
- Action-summary shuffled: 11/30 (36.7%)
- Object-query (5k): 11/30 (36.7%)
- **RGB interpolated checkpoint: 21/30 (70%)** 🏆

**Depth Signal Probe (offline MLP on geometry features)**:
- Contact XYZ: normal RMSE 0.019 vs null 0.031 ✓
- Contact distance: normal RMSE 0.008 vs null 0.025 ✓
- Action XYZ: normal RMSE 0.277 vs null 0.466 ✓
- **Conclusion**: Depth data contains learnable signal, but VLA integration fails to use it causally

### 5.4 Key Challenges Identified

1. **Depth pathway not causally used**: normal ≈ null in most rollouts
2. **Auxiliary supervision insufficient**: geometric labels don't force depth reliance
3. **Data scale limitation**: 5 tasks × 20 demos too small for LIBERO-Plus generalization
4. **Prefix disruption**: appending depth tokens hurts pretrained representations
5. **Camera/initstate variations hard**: where depth should help most, it doesn't yet

### 5.5 Comparison with PointVLA and SpatialVLA

**Similarities**:
- All three augment pretrained VLA with 3D information
- All aim to preserve VLM knowledge
- All focus on spatial reasoning for manipulation

**PointVLA vs DepthVLA**:
- PointVLA: skip-block fusion into transformer layers
- DepthVLA: action-head fusion
- PointVLA shows strong results on long-horizon dynamic tasks (92.5% success)
- DepthVLA struggles with LIBERO-Plus camera/initstate perturbations (14/30 = 46.7%)
- **Key difference**: PointVLA leverages OpenVLA pretrained knowledge (970k demos), we train from scratch on 5 tasks

**SpatialVLA vs DepthVLA**:
- SpatialVLA: Ego3D encoding built into architecture, zero-shot capable
- DepthVLA: modular depth injection, requires task-specific fine-tuning
- SpatialVLA emphasizes spatial position encoding
- DepthVLA emphasizes explicit metric depth + camera geometry
- **Key difference**: SpatialVLA achieves SoTA on LIBERO + Simpler, we focus on robustness analysis with LIBERO-Plus

**Dataset scale impact**:
- **Our training**: 5 tasks × 20 demos = 89 successful demos (11k transitions)
- **PointVLA/SpatialVLA base**: Pretrained on 970k real-world demos (OpenVLA)
- **RLBench methods**: 18+ tasks × 100 demos each = 1800+ demos minimum
- **Impact**: Insufficient task diversity limits depth-action generalization to LIBERO-Plus variations

---

## 6. Summary Table: Method Comparison

| Aspect | PerAct/RVT/DP3 | PointVLA | SpatialVLA | BridgeVLA | DepthVLA (Ours) |
|--------|----------------|----------|------------|-----------|-----------------|
| **3D Representation** | Voxels/views/features | Point clouds | Ego3D encoding | 3D→2D projection | Metric depth + K/T |
| **Base Model** | Trained from scratch | Pretrained VLA | VLA w/ Ego3D | VLM | OpenVLA-7B |
| **Training Data** | RLBench 18×100 | OpenVLA 970k + task | Unknown + LIBERO | RLBench + Bridge | **5 tasks × 20 demos** |
| **VLM Preservation** | N/A | ✓ | Partial | ✓ | ✓ |
| **Integration** | Native | Skip-block | Built-in | 2D heatmaps | Action head |
| **Modular?** | No | Yes | No | Partial | Yes |
| **Zero-shot?** | No | Yes (claimed) | Yes | No | No |
| **Aux Supervision** | Task loss only | Unknown | Unknown | Unknown | 3D geometric targets |
| **Best Result** | RVT: 26% vs PerAct | 92.5% success | Multi-task SoTA | RLBench 88.2% | LIBERO-Plus 70% (RGB) |
| **Depth Usage** | N/A | Not explicitly tested | N/A | N/A | **normal ≈ null** ⚠️ |

---

## 7. Future Directions

Based on comparison with related work and our experimental findings:

### 7.1 Architecture Improvements
1. **Explore skip-block fusion** (inspired by PointVLA):
   - May be more effective than action-head fusion
   - Allows depth to influence intermediate representations
   
2. **Incorporate spatial position encoding** (inspired by SpatialVLA):
   - Ego3D encoding may complement metric depth
   - Could improve spatial reasoning

3. **Hybrid 2D-3D representations** (inspired by BridgeVLA):
   - Project 3D features back to 2D for VLM compatibility
   - Maintain metric information through auxiliary heads

### 7.2 Training Improvements
1. **Scale up data** to LIBERO-Plus variations:
   - **Current limitation**: 5 tasks × 20 demos (89 demos, 11k transitions)
   - **Needed**: 30-50 demos per task across camera/initstate perturbations
   - **Reference**: OpenVLA uses 970k demos, RLBench methods use 1800+ demos
   - Without scaled data, depth-action mappings cannot generalize to perturbations
   
2. **Leverage pretrained VLA knowledge**:
   - Start from OpenVLA pretrained weights (like PointVLA/SpatialVLA)
   - Only fine-tune depth injection modules
   - Our current approach trains from scratch, wasting pretraining

3. **Stronger auxiliary supervision**:
   - Current 5 tasks × 20 demos insufficient
   - Need 30-50 demos per task across camera/initstate perturbations
   
2. **Stronger auxiliary supervision**:
   - True object/target poses (not just visible-geometry proxies)
   - Contrastive loss: normal depth should beat null/shuffle by margin
   - Phase-aware supervision (reach/grasp/transport/place)

3. **Depth-aware data augmentation**:
   - Depth dropout during training
   - Explicit corruption detection
   - Reward models that prefer depth-consistent actions

### 7.3 Evaluation Protocols
1. **Always report ablations**: normal/null/shuffle depth
2. **Multi-trial evaluation**: LIBERO-Plus noise requires ≥3 trials
3. **Failure mode analysis**: which perturbation dimensions fail most?
4. **Attention visualization**: where does model attend in depth/point cloud?

---

## 8. Conclusion

Our DepthVLA work provides a detailed empirical study of integrating metric depth into pretrained VLAs, with extensive ablations revealing critical findings about **why depth integration fails**.

**Key findings**:
1. ✅ **Depth data is informative**: Offline MLP probe achieves normal RMSE 0.019 vs null 0.031 for contact prediction
2. ❌ **VLA fails to use depth causally**: Normal ≈ null in rollouts (14/30 vs 14/30 on LIBERO-Plus)
3. 🔍 **Root cause is NOT data scale**: RGB-only succeeds with same 5-task data (100% clean tasks), DP3 succeeds with 10 demos/task
4. 🎯 **Root cause IS training methodology**: 
   - Auxiliary losses too weak (can be satisfied from RGB/proprio priors)
   - Architecture allows ignoring depth (action head can use RGB only)
   - No explicit depth enforcement (no dropout, no contrastive loss)

**Evidence against data-scale hypothesis**:
- Same 5-task data: Depth probe ✓, RGB-only ✓, DepthVLA ✗
- DP3 succeeds with 10 demos/task (less than our 20 demos/task)
- Data scale explains generalization failure (5→30 tasks), NOT normal≈null

**Evidence for training-method hypothesis**:
- Auxiliary losses (distance_bin, contact_xyz) don't force depth reliance
- Gate init 0.001 actively suppresses depth influence
- Frozen RGB preserves performance but prevents depth integration
- No depth dropout or contrastive training to enforce causality

Compared to concurrent work:
- **PointVLA** and **SpatialVLA** report stronger spatial reasoning (92.5% and SoTA)
  - May have better training methodology (not just more data)
  - Skip-block fusion (PointVLA) or built-in Ego3D (SpatialVLA) may enforce usage better
- **BridgeVLA** achieves 88.2% on RLBench with 2D-3D alignment
  - Heatmap representation may create stronger depth dependency
- Our **LIBERO-Plus robustness analysis** (21/30 = 70% with RGB) exposes camera/initstate brittleness
- **Unique contribution**: Most detailed depth integration failure analysis with normal/null/shuffle ablations

**Critical insight for future work**:
The main bottleneck is **training methodology**, NOT data scale or architecture choice:

**Priority 1 - Fix Training Method**:
1. Depth dropout during training (force reliance when available)
2. Contrastive loss (normal must beat null/shuffle by margin)
3. Phase-aware supervision (reach/grasp/transport need different depth usage)
4. Stronger coupling (depth → auxiliary → action, not parallel paths)

**Priority 2 - Architecture Exploration**:
1. Skip-block fusion (PointVLA-style) - may enforce depth usage better
2. Ego3D encoding (SpatialVLA-style) - built-in spatial reasoning
3. Heatmap outputs (BridgeVLA-style) - explicit spatial grounding

**Priority 3 - Scale Data** (helps generalization, but won't fix causality):
1. Start from OpenVLA pretrained weights (970k demos)
2. Scale to 18+ tasks with camera/initstate variations
3. Include LIBERO-Plus perturbations in training

**Without fixing training method, more data alone won't solve normal ≈ null**

---

## References

[1] M. J. Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," arXiv:2406.09246, 2024.
- https://openvla.github.io/

[2] M. J. Kim, C. Finn, and P. Liang, "Fine-tuning Vision-Language-Action Models: Optimizing Speed and Success," arXiv:2502.19645, 2025.
- https://openvla-oft.github.io/

[3] M. Shridhar, L. Manuelli, and D. Fox, "Perceiver-Actor: A Multi-task Transformer for Robotic Manipulation," CoRL 2023.
- https://peract.github.io/

[4] A. Goyal et al., "RVT: Robotic View Transformer for 3D Object Manipulation," CoRL 2023.
- https://robotic-view-transformer.github.io/

[5] Y. Ze et al., "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations," arXiv:2403.03954, 2024.
- https://arxiv.org/abs/2403.03954

[6] C. Li et al., "PointVLA: Injecting the 3D World into Vision-Language-Action Models," IEEE RA-L, 2026.
- https://arxiv.org/abs/2503.07511

[7] D. Qu et al., "SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model," arXiv:2501.15830, 2025.
- https://arxiv.org/abs/2501.15830

[8] P. Li et al., "BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models," NeurIPS 2025.
- https://arxiv.org/abs/2506.07961

[9] B. Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning," NeurIPS 2023.

[10] S. Fei et al., "LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models," arXiv:2510.13626, 2025.
- https://github.com/sylvestf/LIBERO-plus
