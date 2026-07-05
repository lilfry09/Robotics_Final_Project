"""
Test heatmap generation on real LIBERO-Plus symbolic RGB-D data.

This script:
1. Loads one episode from the existing symbolic HDF5
2. Generates object/target heatmaps
3. Visualizes overlay on RGB images
4. Validates projection correctness
"""

import sys
import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prismatic.models.heatmap_generator import (
    create_heatmap_labels_batch,
    visualize_heatmap_overlay
)


def load_episode_data(hdf5_path, demo_idx=0):
    """Load one episode from HDF5."""
    print(f"Loading data from: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        demo_keys = list(f['data'].keys())
        demo_key = demo_keys[demo_idx]
        print(f"  Using demo: {demo_key}")

        demo = f['data'][demo_key]

        # Load necessary fields
        data = {
            'agentview_rgb': demo['obs']['agentview_rgb'][:],
            'manipulated_object_pos': demo['obs']['manipulated_object_pos'][:],
            'target_pos': demo['obs']['target_pos'][:],
            'agentview_K': demo['obs']['agentview_K'][:],
            'agentview_T_camera_to_base': demo['obs']['agentview_T_camera_to_base'][:],
        }

        # Get metadata
        data['task_name'] = demo.attrs['task_name'] if 'task_name' in demo.attrs else 'unknown'
        data['object_name'] = demo.attrs['manipulated_object_name'] if 'manipulated_object_name' in demo.attrs else 'unknown'
        data['target_name'] = demo.attrs['target_object_name'] if 'target_object_name' in demo.attrs else 'unknown'

    print(f"  Loaded {len(data['agentview_rgb'])} timesteps")
    return data


def test_heatmap_generation(data, num_samples=5):
    """Generate and visualize heatmaps for a few timesteps."""

    T = len(data['agentview_rgb'])

    # Sample timesteps
    sample_indices = np.linspace(0, T-1, num_samples, dtype=int)

    print(f"\nGenerating heatmaps for {num_samples} timesteps...")

    results = []

    for idx in sample_indices:
        # Prepare batch (single sample)
        object_pos = torch.from_numpy(data['manipulated_object_pos'][idx]).float().unsqueeze(0)  # (1, 3)
        target_pos = torch.from_numpy(data['target_pos'][idx]).float().unsqueeze(0)  # (1, 3)

        # Camera params (time-varying in symbolic Plus data)
        K_batch = torch.from_numpy(data['agentview_K'][idx]).float().unsqueeze(0)
        T_batch = torch.from_numpy(data['agentview_T_camera_to_base'][idx]).float().unsqueeze(0)

        # Generate heatmaps
        heatmap_dict = create_heatmap_labels_batch(
            object_pos,
            target_pos,
            K_batch,
            T_batch,
            image_height=256,
            image_width=256,
            sigma=15.0
        )

        # Get RGB image
        rgb_image = data['agentview_rgb'][idx]  # (256, 256, 3)

        # Store results
        results.append({
            'timestep': idx,
            'rgb_image': rgb_image,
            'object_heatmap': heatmap_dict['object_heatmap'][0].numpy(),  # (256, 256)
            'target_heatmap': heatmap_dict['target_heatmap'][0].numpy(),
            'object_valid': heatmap_dict['object_valid'][0].item(),
            'target_valid': heatmap_dict['target_valid'][0].item(),
            'object_pos': object_pos[0].numpy(),
            'target_pos': target_pos[0].numpy(),
        })

        print(f"  t={idx}: object_valid={results[-1]['object_valid']}, "
              f"target_valid={results[-1]['target_valid']}")

    return results


def visualize_results(results, save_dir=None):
    """Create visualization grid."""

    num_samples = len(results)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, result in enumerate(results):
        rgb = result['rgb_image']
        obj_hm = result['object_heatmap']
        tgt_hm = result['target_heatmap']

        # Column 0: RGB
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"t={result['timestep']}: RGB")
        axes[i, 0].axis('off')

        # Column 1: Object heatmap overlay
        if result['object_valid']:
            overlay_obj = visualize_heatmap_overlay(rgb, obj_hm, alpha=0.5, colormap='hot')
            axes[i, 1].imshow(overlay_obj)
            axes[i, 1].set_title(f"Object Heatmap\n(max={obj_hm.max():.3f})")
        else:
            axes[i, 1].imshow(rgb)
            axes[i, 1].set_title("Object (invalid)")
        axes[i, 1].axis('off')

        # Column 2: Target heatmap overlay
        if result['target_valid']:
            overlay_tgt = visualize_heatmap_overlay(rgb, tgt_hm, alpha=0.5, colormap='cool')
            axes[i, 2].imshow(overlay_tgt)
            axes[i, 2].set_title(f"Target Heatmap\n(max={tgt_hm.max():.3f})")
        else:
            axes[i, 2].imshow(rgb)
            axes[i, 2].set_title("Target (invalid)")
        axes[i, 2].axis('off')

        # Column 3: Combined overlay
        combined_rgb = rgb.copy()
        if result['object_valid']:
            combined_rgb = visualize_heatmap_overlay(combined_rgb, obj_hm, alpha=0.3, colormap='hot')
        if result['target_valid']:
            combined_rgb = visualize_heatmap_overlay(combined_rgb, tgt_hm, alpha=0.3, colormap='cool')
        axes[i, 3].imshow(combined_rgb)
        axes[i, 3].set_title("Combined\n(red=obj, blue=tgt)")
        axes[i, 3].axis('off')

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'heatmap_test_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    # Path to symbolic Plus RGB-D data - use first available HDF5
    dataset_dir = "/root/autodl-tmp/LIBERO/libero/datasets/libero_spatial_plus_rgbd_30tasks_2demos"

    import glob
    hdf5_files = glob.glob(os.path.join(dataset_dir, "*.hdf5"))

    if not hdf5_files:
        print(f"❌ No HDF5 files found in: {dataset_dir}")
        return

    hdf5_path = hdf5_files[0]  # Use first file

    print("=" * 70)
    print("Testing Heatmap Generation on Real LIBERO-Plus Data")
    print("=" * 70)

    # Load one episode
    data = load_episode_data(hdf5_path, demo_idx=0)

    # Print metadata if available
    if 'task_name' in data:
        print(f"\nTask: {data['task_name']}")
    if 'object_name' in data:
        print(f"Object: {data['object_name']}")
    if 'target_name' in data:
        print(f"Target: {data['target_name']}")

    # Generate heatmaps
    results = test_heatmap_generation(data, num_samples=5)

    # Compute statistics
    object_valid_ratio = sum(r['object_valid'] for r in results) / len(results)
    target_valid_ratio = sum(r['target_valid'] for r in results) / len(results)

    avg_object_peak = np.mean([r['object_heatmap'].max() for r in results if r['object_valid']])
    avg_target_peak = np.mean([r['target_heatmap'].max() for r in results if r['target_valid']])

    print(f"\n" + "=" * 70)
    print("Statistics:")
    print(f"  Object valid ratio: {object_valid_ratio:.1%}")
    print(f"  Target valid ratio: {target_valid_ratio:.1%}")
    print(f"  Avg object heatmap peak: {avg_object_peak:.3f}")
    print(f"  Avg target heatmap peak: {avg_target_peak:.3f}")
    print("=" * 70)

    # Visualize
    save_dir = "/root/autodl-tmp/openvla-oft/experiments/heatmap_test"
    visualize_results(results, save_dir=save_dir)

    print("\n✓ Test completed successfully!")
    print("\nNext steps:")
    print("1. Check the visualization to verify heatmaps align with objects")
    print("2. If alignment looks good, proceed to implement HeatmapPredictionHead")
    print("3. Train heatmap prediction module (Stage A)")


if __name__ == "__main__":
    main()
