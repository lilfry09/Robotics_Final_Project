#!/bin/bash
#
# Quick heatmap training wrapper
# Uses existing RGB checkpoint and adds minimal heatmap fusion
#

set -e

# Activate environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate depthvla

cd /root/autodl-tmp/openvla-oft

echo "=========================================="
echo "BridgeVLA Heatmap Training"
echo "=========================================="

# Configuration
VLA_PATH="/root/autodl-tmp/hf-cache/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f"
RESUME_FROM="/root/autodl-tmp/openvla-oft/runs_depthvla_plus_rgb_mix_phaseA/rgb-only-anchor-mix3k-interp-alpha0.5"
RUN_DIR="runs_bridgevla_heatmap"
RUN_ID="heatmap_simple_alpha0.1_1500steps"

# Mixed dataset (70% 5-task, 30% Plus)
DATASET_5TASK="/root/autodl-tmp/LIBERO/libero/datasets/libero_spatial_rgbd_5tasks_20demos"
DATASET_PLUS="/root/autodl-tmp/LIBERO/libero/datasets/libero_spatial_plus_rgbd_30tasks_2demos"

echo ""
echo "Config:"
echo "  Resume from: $RESUME_FROM"
echo "  Max steps: 1500"
echo "  Save interval: 500"
echo "  Heatmap alpha: 0.1"
echo ""

# Note: This is a simplified wrapper
# The actual heatmap integration requires modifying finetune_depthvla.py
# For now, we'll run RGB-only training with the best checkpoint

echo "⚠️  WARNING: Full heatmap integration requires code modification"
echo "    For rapid implementation, we'll prepare the training command"
echo ""

cat > /tmp/heatmap_training_command.sh << 'EOF'
# This would be the training command once heatmap support is added to finetune_depthvla.py

python vla-scripts/finetune_depthvla.py \
  --vla_path /root/autodl-tmp/hf-cache/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --rgbd_data_dir /root/autodl-tmp/LIBERO/libero/datasets \
  --dataset_name libero_spatial_rgbd_5tasks_20demos \
  --use_depth False \
  --use_heatmap True \
  --heatmap_alpha_init 0.1 \
  --freeze_heatmap_alpha False \
  --resume_components_from /root/autodl-tmp/openvla-oft/runs_depthvla_plus_rgb_mix_phaseA/rgb-only-anchor-mix3k-interp-alpha0.5 \
  --freeze_vla_lora True \
  --freeze_proprio_projector True \
  --freeze_action_head_base True \
  --max_steps 1500 \
  --save_interval 500 \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --run_root_dir runs_bridgevla_heatmap \
  --run_id heatmap_simple_alpha0.1_1500steps
EOF

echo "Training command saved to /tmp/heatmap_training_command.sh"
echo ""
echo "Next steps:"
echo "1. Add heatmap support flags to DepthFinetuneConfig"
echo "2. Modify LiberoRGBDHDF5Dataset to generate heatmaps"
echo "3. Add heatmap fusion to forward pass"
echo ""
