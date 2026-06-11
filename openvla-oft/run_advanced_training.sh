#!/bin/bash
# DepthVLA Advanced Training - Beat RGB-Only and SOTA
# Three-stage training: Pretrain → Integrate → Multi-dataset finetune

set -e

# ============================================================================
# Configuration
# ============================================================================

EXP_NAME="depthvla_advanced_v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="/root/autodl-tmp/openvla-oft"
RESULTS_DIR="${BASE_DIR}/experiments/advanced_${TIMESTAMP}"

# Environment
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/root/autodl-tmp/hf-cache
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

PYTHON=/root/miniconda3/envs/depthvla/bin/python

# Paths
RGB_CHECKPOINT="${BASE_DIR}/runs_depthvla_stage2/openvla-7b+libero_spatial_rgbd_5tasks_20demos+rgb-only+b1+lr-0.0001+lora-r4+dropout-0.0"
LIBERO_DATA="/root/autodl-tmp/LIBERO/libero/datasets/libero_spatial_rgbd_5tasks_20demos"

mkdir -p ${RESULTS_DIR}

echo "============================================"
echo "DepthVLA Advanced Training"
echo "Experiment: ${EXP_NAME}"
echo "Timestamp: ${TIMESTAMP}"
echo "============================================"
echo ""

# ============================================================================
# Stage 1: Quick Validation (5k steps, same 5 tasks)
# ============================================================================

echo "Stage 1: Quick Validation with Advanced Components"
echo "---------------------------------------------------"

STAGE1_CONFIG="${RESULTS_DIR}/stage1_config.yaml"
cat > ${STAGE1_CONFIG} <<EOF
# Stage 1: Validate advanced training method on 5 tasks
experiment_name: ${EXP_NAME}_stage1_validation
objective: Verify normal > null with new training method

# Key improvements
depth_dropout_initial: 0.5
depth_dropout_final: 0.2
depth_dropout_steps: 5000
depth_dropout_schedule: cosine

contrastive_margin: 0.05
contrastive_weight: 0.3
contrastive_hierarchy_weights: [1.0, 0.5, 0.3]

spatial_weight: 0.2
spatial_coarse_weight: 0.3
spatial_medium_weight: 0.5
spatial_fine_weight: 0.2

# Architecture
depth_integration_mode: depth_action_summary_aux
depth_grid_size: 8  # Increase from 4
depth_hidden_dim: 512  # Increase from 256
depth_action_fusion_gate_init: 1.0

# Training
max_steps: 5000
batch_size: 4  # Increase for better contrastive
gradient_accumulation_steps: 2
learning_rate: 5e-5
save_freq: 1000

# Freeze strategy
freeze_vla_lora: true
freeze_proprio_projector: true
freeze_action_head_base: true

# Success criteria
target_normal_success: 14/15  # At least 93%
target_null_success: 11/15    # At most 73%
target_gap: 3                 # Normal - Null ≥ 3 tasks
EOF

echo "Training Stage 1..."
${PYTHON} vla-scripts/finetune_depthvla_advanced.py \
  --config ${STAGE1_CONFIG} \
  --data_root ${LIBERO_DATA} \
  --resume_from_rgb ${RGB_CHECKPOINT} \
  --output_dir ${RESULTS_DIR}/stage1 \
  2>&1 | tee ${RESULTS_DIR}/stage1_training.log

echo ""
echo "Stage 1 Training Complete!"
echo "Running ablations..."

# Evaluate Stage 1
for mode in normal null shuffle; do
  echo "Evaluating: ${mode}"
  ${PYTHON} experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ${RESULTS_DIR}/stage1/checkpoint_final \
    --task_ids 0,1,2,7,9 \
    --num_trials_per_task 3 \
    --depth_ablation_mode ${mode} \
    --run_id_note stage1_${mode} \
    2>&1 | tee ${RESULTS_DIR}/stage1_eval_${mode}.log
done

echo ""
echo "Stage 1 Ablations Complete!"
echo ""

# Parse results
NORMAL_RESULT=$(grep "overall success rate" ${RESULTS_DIR}/stage1_eval_normal.log | tail -1)
NULL_RESULT=$(grep "overall success rate" ${RESULTS_DIR}/stage1_eval_null.log | tail -1)
SHUFFLE_RESULT=$(grep "overall success rate" ${RESULTS_DIR}/stage1_eval_shuffle.log | tail -1)

echo "Stage 1 Results:"
echo "  Normal:  ${NORMAL_RESULT}"
echo "  Null:    ${NULL_RESULT}"
echo "  Shuffle: ${SHUFFLE_RESULT}"
echo ""

# Check if we should proceed
echo "Checking success criteria..."
# TODO: Add automatic parsing and decision logic
echo "Manual check: Does Normal > Null by ≥3 tasks?"
read -p "Continue to Stage 2? (y/n): " continue_stage2

if [ "$continue_stage2" != "y" ]; then
  echo "Stopping. Please analyze Stage 1 results and tune hyperparameters."
  exit 0
fi

# ============================================================================
# Stage 2: Scale to LIBERO-Plus (if Stage 1 succeeds)
# ============================================================================

echo ""
echo "Stage 2: Scale to LIBERO-Plus Robustness"
echo "---------------------------------------------------"

STAGE2_CONFIG="${RESULTS_DIR}/stage2_config.yaml"
cat > ${STAGE2_CONFIG} <<EOF
# Stage 2: Evaluate on 30-task robustness probe
experiment_name: ${EXP_NAME}_stage2_libero_plus
objective: Beat RGB-only baseline (17/30)

# Use best config from Stage 1
depth_dropout_initial: 0.5
depth_dropout_final: 0.2
contrastive_weight: 0.3
spatial_weight: 0.2

# Evaluation
eval_tasks: libero_plus_30_probe
num_trials_per_task: 1

# Success criteria
target_success: 22/30  # Beat RGB 17/30 by 5 tasks
EOF

echo "Evaluating Stage 1 checkpoint on LIBERO-Plus..."

# Get LIBERO-Plus task IDs
LIBERO_PLUS_TASKS="1725,1726,1727,1728,1729,1730,1731,1732,1733,1734,1735,1736,1737,1738,1739,1740,1741,1742,1743,1744,1745,1746,1747,1748,1749,1750,1751,1752,1753,287"

${PYTHON} experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint ${RESULTS_DIR}/stage1/checkpoint_final \
  --task_suite_name libero_spatial \
  --task_ids ${LIBERO_PLUS_TASKS} \
  --num_trials_per_task 1 \
  --depth_ablation_mode normal \
  --run_id_note stage2_libero_plus_normal \
  2>&1 | tee ${RESULTS_DIR}/stage2_libero_plus_normal.log

# Null depth comparison
${PYTHON} experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint ${RESULTS_DIR}/stage1/checkpoint_final \
  --task_suite_name libero_spatial \
  --task_ids ${LIBERO_PLUS_TASKS} \
  --num_trials_per_task 1 \
  --depth_ablation_mode null \
  --run_id_note stage2_libero_plus_null \
  2>&1 | tee ${RESULTS_DIR}/stage2_libero_plus_null.log

echo ""
echo "Stage 2 LIBERO-Plus Evaluation Complete!"
echo ""

# Parse LIBERO-Plus results
LIBERO_PLUS_NORMAL=$(grep "overall success rate" ${RESULTS_DIR}/stage2_libero_plus_normal.log | tail -1)
LIBERO_PLUS_NULL=$(grep "overall success rate" ${RESULTS_DIR}/stage2_libero_plus_null.log | tail -1)

echo "LIBERO-Plus Results:"
echo "  Normal: ${LIBERO_PLUS_NORMAL}"
echo "  Null:   ${LIBERO_PLUS_NULL}"
echo "  RGB-only baseline: 17/30 (56.7%)"
echo "  Target: 22/30 (73.3%)"
echo ""

# ============================================================================
# Stage 3: Multi-Dataset Training (if Stage 2 succeeds)
# ============================================================================

echo "Stage 3: Multi-Dataset Fine-tuning"
echo "---------------------------------------------------"
echo "Note: Requires RLBench, Calvin datasets"
echo "This stage is optional and can be run separately"
echo ""

cat > ${RESULTS_DIR}/stage3_plan.md <<EOF
# Stage 3: Multi-Dataset Training Plan

## Objective
Beat SOTA on multiple benchmarks:
- LIBERO-Plus: 24/30 (80%)
- RLBench: 90%+
- Calvin: 85%+

## Requirements
1. Download RLBench dataset
2. Download Calvin dataset
3. Implement multi-dataset loader
4. Train for 30k steps on mixed data

## Estimated Resources
- GPU: 1x A100 (40GB)
- Time: ~3-4 days
- Disk: ~200GB for all datasets

## Steps
1. Download datasets (see data_download.sh)
2. Implement DatasetMixer in finetune_depthvla_advanced.py
3. Run multi-dataset training
4. Evaluate on all benchmarks

## Expected Performance
With all improvements:
- LIBERO-Plus: 22-24/30
- RLBench: 88-90%
- Calvin: 80-85%

This would match or beat:
- PointVLA (92.5% on long-horizon)
- BridgeVLA (88.2% on RLBench)
- SpatialVLA (SoTA on LIBERO)
EOF

echo "Stage 3 plan saved to: ${RESULTS_DIR}/stage3_plan.md"
echo ""

# ============================================================================
# Final Summary
# ============================================================================

echo "============================================"
echo "Experiment Complete!"
echo "============================================"
echo ""

cat > ${RESULTS_DIR}/SUMMARY.md <<EOF
# DepthVLA Advanced Training - Results Summary

**Experiment**: ${EXP_NAME}
**Timestamp**: ${TIMESTAMP}

## Stage 1: Validation (5 tasks, 5k steps)

### Configuration
- Curriculum depth dropout: 0.5 → 0.2
- Multi-level contrastive: margin 0.05, weight 0.3
- Hierarchical spatial supervision: weight 0.2
- Grid size: 8x8 (increased from 4x4)
- Hidden dim: 512 (increased from 256)

### Results (Clean Tasks 0,1,2,7,9, 3 trials)
- Normal:  ${NORMAL_RESULT}
- Null:    ${NULL_RESULT}
- Shuffle: ${SHUFFLE_RESULT}

### Analysis
- [ ] Normal > Null by ≥3 tasks? (Success criterion)
- [ ] Normal > Shuffle by ≥1 task?
- [ ] Normal ≥ 13/15 (preserve RGB-only quality)?

## Stage 2: LIBERO-Plus Robustness (30 tasks, 1 trial)

### Results
- Normal: ${LIBERO_PLUS_NORMAL}
- Null:   ${LIBERO_PLUS_NULL}

### Baselines
- RGB-only: 17/30 (56.7%)
- Previous best depth: 14/30 (46.7%)

### Analysis
- [ ] Normal > 17/30 (beat RGB-only)?
- [ ] Normal > 20/30 (substantial improvement)?
- [ ] Normal - Null ≥ 3 (clear depth usage)?

## Key Improvements Over Baseline

1. **Curriculum Dropout**: Forces early RGB learning, gradual depth integration
2. **Multi-Level Contrastive**: Enforces depth usage at action, hidden, residual levels
3. **Hierarchical Supervision**: Multi-scale 3D targets (occupancy → vectors → contacts)
4. **Architecture Upgrades**: 8x8 grid, 512 hidden dim

## Next Steps

### If Stage 1 Succeeded (Normal > Null)
✓ Proceed to Stage 2 LIBERO-Plus evaluation
✓ If Stage 2 beats RGB-only → publish results

### If Stage 1 Failed (Normal ≈ Null)
- Increase contrastive_weight (0.3 → 0.5)
- Increase dropout_initial (0.5 → 0.7)
- Add adversarial null training
- Check implementation bugs

### If Stage 2 Failed (Normal < RGB-only)
- Depth helps on clean tasks but hurts on robustness
- Need more diverse training data
- Implement Stage 3 multi-dataset training

### Stage 3 (Optional)
- Multi-dataset pretraining (RLBench, Calvin, NYU Depth)
- Hard negative mining
- Cross-dataset evaluation

## Files Generated
- Training logs: stage1_training.log
- Eval logs: stage1_eval_{normal,null,shuffle}.log
- LIBERO-Plus logs: stage2_libero_plus_{normal,null}.log
- Config files: stage{1,2,3}_config.yaml
- This summary: SUMMARY.md

## Reproducibility
\`\`\`bash
# Rerun entire pipeline
cd ${BASE_DIR}
./run_advanced_training.sh

# Or run stages individually
python vla-scripts/finetune_depthvla_advanced.py --config stage1_config.yaml
python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint ...
\`\`\`
EOF

echo "Results saved to: ${RESULTS_DIR}/SUMMARY.md"
echo ""
echo "View full summary:"
echo "  cat ${RESULTS_DIR}/SUMMARY.md"
echo ""
echo "Next steps:"
echo "1. Check if Normal > Null by ≥3 tasks in Stage 1"
echo "2. If yes, check if Normal > 17/30 on LIBERO-Plus"
echo "3. If both yes, you've beaten RGB-only! 🎉"
echo ""
echo "For Stage 3 multi-dataset training:"
echo "  See ${RESULTS_DIR}/stage3_plan.md"
echo ""
