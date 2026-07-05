"""
Critical Validation: Normal vs Null vs Shuffle Depth

This is THE test that proves depth is being used causally.

Expected results:
- Normal depth: High PSNR (>25 dB), low peak distance (<5 pixels)
- Null depth: Low PSNR (<15 dB), random predictions
- Shuffle depth: Low PSNR (<15 dB), corrupted predictions

If normal >> null, we've proven depth is causally used!
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import sys; sys.path.insert(0, "vla-scripts"); from train_quick_heatmap import QuickHeatmapDataset, collate_fn
from prismatic.models.depth_encoder import LightweightDepthTokenEncoder
from prismatic.models.minimal_heatmap_head import MinimalHeatmapHead
from prismatic.models.heatmap_generator import compute_heatmap_metrics, visualize_heatmap_overlay


def corrupt_depth(depth, mode='null'):
    """Corrupt depth for ablation."""
    if mode == 'null':
        return torch.zeros_like(depth)
    elif mode == 'shuffle':
        B, H, W, C = depth.shape
        corrupted = depth.clone()
        for b in range(B):
            flat = corrupted[b].flatten()
            indices = torch.randperm(len(flat))
            corrupted[b] = flat[indices].reshape(H, W, C)
        return corrupted
    else:
        return depth


def evaluate_ablation(checkpoint_path, data_dir, mode='normal'):
    """Evaluate with normal/null/shuffle depth."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    depth_encoder = LightweightDepthTokenEncoder(llm_dim=1024, grid_size=8, hidden_dim=512).to(device)
    heatmap_head = MinimalHeatmapHead(depth_token_dim=1024, hidden_dim=256).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    depth_encoder.load_state_dict(checkpoint['depth_encoder'])
    heatmap_head.load_state_dict(checkpoint['heatmap_head'])

    depth_encoder.eval()
    heatmap_head.eval()

    # Dataset
    import sys; sys.path.insert(0, "vla-scripts"); from train_quick_heatmap import QuickHeatmapDataset, collate_fn
    dataset = QuickHeatmapDataset(data_dir, max_episodes=12)  # All 12 episodes
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    print(f"\nEvaluating with mode: {mode.upper()}")
    print("="*70)

    all_obj_metrics = []
    all_tgt_metrics = []

    sample_visualizations = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            agentview_depth = batch['agentview_depth'].to(device)
            eye_in_hand_depth = batch['eye_in_hand_depth'].to(device)

            # Corrupt depth based on mode
            if mode == 'null':
                agentview_depth = corrupt_depth(agentview_depth, 'null')
                eye_in_hand_depth = corrupt_depth(eye_in_hand_depth, 'null')
            elif mode == 'shuffle':
                agentview_depth = corrupt_depth(agentview_depth, 'shuffle')
                eye_in_hand_depth = corrupt_depth(eye_in_hand_depth, 'shuffle')

            agentview_K = batch['agentview_K'].to(device)
            agentview_T = batch['agentview_T'].to(device)
            eye_in_hand_K = batch['eye_in_hand_K'].to(device)
            eye_in_hand_T = batch['eye_in_hand_T'].to(device)

            object_hm_gt = batch['object_heatmap_gt'].to(device)
            target_hm_gt = batch['target_heatmap_gt'].to(device)
            object_valid = batch['object_valid'].to(device)
            target_valid = batch['target_valid'].to(device)

            # Predict
            B = agentview_depth.shape[0]
            depth_values = torch.cat([
                agentview_depth.squeeze(-1).unsqueeze(1),
                eye_in_hand_depth.squeeze(-1).unsqueeze(1)
            ], dim=1)
            depth_K = torch.stack([agentview_K, eye_in_hand_K], dim=1)
            depth_T = torch.stack([agentview_T, eye_in_hand_T], dim=1)
            depth_tokens = depth_encoder(depth_values, depth_K, depth_T)
            object_hm_pred, target_hm_pred = heatmap_head(depth_tokens)

            # Metrics
            obj_metrics = compute_heatmap_metrics(
                object_hm_pred.squeeze(1), object_hm_gt.squeeze(1), object_valid
            )
            tgt_metrics = compute_heatmap_metrics(
                target_hm_pred.squeeze(1), target_hm_gt.squeeze(1), target_valid
            )

            all_obj_metrics.append(obj_metrics)
            all_tgt_metrics.append(tgt_metrics)

            # Save sample for visualization
            if len(sample_visualizations) < 3:
                sample_visualizations.append({
                    'object_pred': object_hm_pred[0, 0].cpu().numpy(),
                    'object_gt': object_hm_gt[0, 0].cpu().numpy(),
                    'target_pred': target_hm_pred[0, 0].cpu().numpy(),
                    'target_gt': target_hm_gt[0, 0].cpu().numpy(),
                })

    # Aggregate results
    results = {
        'mode': mode,
        'object_psnr': np.mean([m['psnr'] for m in all_obj_metrics if m['num_valid'] > 0]),
        'target_psnr': np.mean([m['psnr'] for m in all_tgt_metrics if m['num_valid'] > 0]),
        'object_peak_dist': np.mean([m['peak_distance'] for m in all_obj_metrics if m['num_valid'] > 0]),
        'target_peak_dist': np.mean([m['peak_distance'] for m in all_tgt_metrics if m['num_valid'] > 0]),
        'object_mse': np.mean([m['mse'] for m in all_obj_metrics if m['num_valid'] > 0]),
        'target_mse': np.mean([m['mse'] for m in all_tgt_metrics if m['num_valid'] > 0]),
    }

    return results, sample_visualizations


def create_comparison_table(results_dict):
    """Create comparison table."""
    print("\n" + "="*90)
    print("HEATMAP PREDICTION ABLATION RESULTS")
    print("="*90)
    print(f"{'Mode':<15} {'Obj PSNR':>12} {'Tgt PSNR':>12} {'Obj Peak Dist':>15} {'Tgt Peak Dist':>15}")
    print("-"*90)

    for mode in ['normal', 'null', 'shuffle']:
        if mode in results_dict:
            r = results_dict[mode]
            print(f"{mode.upper():<15} {r['object_psnr']:>10.2f} dB {r['target_psnr']:>10.2f} dB "
                  f"{r['object_peak_dist']:>12.2f} px {r['target_peak_dist']:>12.2f} px")

    print("="*90)

    # Interpretation
    print("\n" + "="*90)
    print("INTERPRETATION")
    print("="*90)

    if 'normal' in results_dict and 'null' in results_dict:
        normal = results_dict['normal']
        null = results_dict['null']

        psnr_improvement = normal['object_psnr'] - null['object_psnr']
        peak_improvement = null['object_peak_dist'] - normal['object_peak_dist']

        print(f"\nObject Heatmap:")
        print(f"  PSNR improvement (normal vs null): {psnr_improvement:+.2f} dB")
        print(f"  Peak distance improvement: {peak_improvement:+.2f} pixels")

        # Success criteria
        print(f"\n✓ SUCCESS CRITERIA:")

        success = True
        if normal['object_psnr'] > 25:
            print(f"  ✓ Normal PSNR > 25 dB: {normal['object_psnr']:.2f} dB")
        else:
            print(f"  ✗ Normal PSNR < 25 dB: {normal['object_psnr']:.2f} dB")
            success = False

        if null['object_psnr'] < 15:
            print(f"  ✓ Null PSNR < 15 dB: {null['object_psnr']:.2f} dB")
        else:
            print(f"  ✗ Null PSNR > 15 dB: {null['object_psnr']:.2f} dB")
            success = False

        if normal['object_peak_dist'] < 5:
            print(f"  ✓ Normal peak distance < 5 pixels: {normal['object_peak_dist']:.2f} px")
        else:
            print(f"  ✗ Normal peak distance > 5 pixels: {normal['object_peak_dist']:.2f} px")
            success = False

        if psnr_improvement > 10:
            print(f"  ✓ PSNR separation > 10 dB: {psnr_improvement:.2f} dB")
        else:
            print(f"  ✗ PSNR separation < 10 dB: {psnr_improvement:.2f} dB")
            success = False

        print("\n" + "="*90)
        if success:
            print("✅ VALIDATION PASSED: Depth information is being used causally!")
            print("   → Normal depth produces significantly better heatmaps than null depth")
            print("   → This proves the model CANNOT bypass depth information")
            print("   → Ready to proceed with action conditioning")
        else:
            print("❌ VALIDATION FAILED: Depth is not being used effectively")
            print("   → Need to debug depth encoder or increase training")
        print("="*90)

    return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='/root/autodl-tmp/openvla-oft/runs_quick_heatmap/checkpoint_best.pt')
    parser.add_argument('--data_dir', type=str,
                        default='/root/autodl-tmp/LIBERO/libero/datasets/libero_spatial_plus_rgbd_30tasks_2demos')
    parser.add_argument('--save_dir', type=str,
                        default='/root/autodl-tmp/openvla-oft/experiments/heatmap_ablation')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("Please train the model first with:")
        print("  python vla-scripts/train_quick_heatmap.py")
        return

    print("="*90)
    print("HEATMAP CAUSALITY TEST")
    print("="*90)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_dir}")

    # Run ablations
    results_dict = {}

    for mode in ['normal', 'null', 'shuffle']:
        results, samples = evaluate_ablation(args.checkpoint, args.data_dir, mode=mode)
        results_dict[mode] = results

        print(f"\n{mode.upper()}: Object PSNR = {results['object_psnr']:.2f} dB, "
              f"Peak dist = {results['object_peak_dist']:.2f} px")

    # Create table
    create_comparison_table(results_dict)

    # Save results
    import json
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✓ Results saved to: {args.save_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
