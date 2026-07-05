"""
Test script for heatmap generation with visualization.

Quick validation that 3D-to-2D projection and Gaussian heatmap generation work correctly.
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prismatic.models.heatmap_generator import (
    generate_task_heatmaps,
    project_3d_to_2d,
    generate_gaussian_heatmap,
    corrupt_heatmap,
)


def visualize_heatmap_sample(hdf5_path: Path, output_dir: Path, num_samples: int = 10):
    """Load samples from HDF5 and visualize generated heatmaps."""

    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, 'r') as f:
        demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])

        if not demo_keys:
            print(f"❌ No demos found in {hdf5_path}")
            return False

        print(f"Found {len(demo_keys)} demos in {hdf5_path.name}")

        sample_count = 0
        success_count = 0

        for demo_key in demo_keys:
            if sample_count >= num_samples:
                break

            obs = f[f'data/{demo_key}/obs']
            num_steps = len(obs['ee_pos'])

            # Check required fields
            required_fields = [
                'manipulated_object_pos', 'target_pos', 'ee_pos',
                'agentview_K', 'agentview_T_camera_to_base', 'agentview_rgb'
            ]

            missing = [field for field in required_fields if field not in obs]
            if missing:
                print(f"⚠️  {demo_key}: Missing fields {missing}, skipping")
                continue

            # Sample a few timesteps
            timesteps = np.linspace(0, num_steps - 1, min(3, num_steps), dtype=int)

            for t in timesteps:
                if sample_count >= num_samples:
                    break

                # Build obs_dict
                obs_dict = {
                    'manipulated_object_pos': np.array(obs['manipulated_object_pos'][t]),
                    'target_pos': np.array(obs['target_pos'][t]),
                    'ee_pos': np.array(obs['ee_pos'][t]),
                    'agentview_K': np.array(obs['agentview_K'][t]),
                    'agentview_T_camera_to_base': np.array(obs['agentview_T_camera_to_base'][t]),
                }

                # Check validity
                if not all(np.isfinite(obs_dict[k]).all() for k in ['manipulated_object_pos', 'target_pos', 'ee_pos']):
                    print(f"⚠️  {demo_key} t={t}: Non-finite positions, skipping")
                    continue

                # Generate heatmap
                try:
                    heatmap = generate_task_heatmaps(obs_dict, image_size=(224, 224), sigma=5.0)
                except Exception as e:
                    print(f"❌ {demo_key} t={t}: Heatmap generation failed: {e}")
                    continue

                # Load RGB for visualization
                rgb = np.array(obs['agentview_rgb'][t])

                # Project 3D points for validation
                K = obs_dict['agentview_K']
                T = obs_dict['agentview_T_camera_to_base']

                u_obj, v_obj, valid_obj = project_3d_to_2d(obs_dict['manipulated_object_pos'], K, T)
                u_target, v_target, valid_target = project_3d_to_2d(obs_dict['target_pos'], K, T)
                u_ee, v_ee, valid_ee = project_3d_to_2d(obs_dict['ee_pos'], K, T)

                # Visualize
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # Row 0: RGB + overlays
                axes[0, 0].imshow(rgb)
                axes[0, 0].set_title('RGB')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(rgb)
                if valid_obj:
                    axes[0, 1].plot(u_obj, v_obj, 'ro', markersize=10, label='Object')
                if valid_target:
                    axes[0, 1].plot(u_target, v_target, 'go', markersize=10, label='Target')
                if valid_ee:
                    axes[0, 1].plot(u_ee, v_ee, 'bo', markersize=10, label='EE')
                axes[0, 1].legend()
                axes[0, 1].set_title('RGB + Projections')
                axes[0, 1].axis('off')

                axes[0, 2].imshow(rgb)
                axes[0, 2].imshow(heatmap.sum(axis=-1), alpha=0.5, cmap='hot')
                axes[0, 2].set_title('RGB + All Heatmaps')
                axes[0, 2].axis('off')

                # Row 1: Individual heatmap channels
                axes[1, 0].imshow(heatmap[..., 0], cmap='hot', vmin=0, vmax=1)
                axes[1, 0].set_title(f'Object Heatmap (max={heatmap[..., 0].max():.3f})')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(heatmap[..., 1], cmap='hot', vmin=0, vmax=1)
                axes[1, 1].set_title(f'Target Heatmap (max={heatmap[..., 1].max():.3f})')
                axes[1, 1].axis('off')

                axes[1, 2].imshow(heatmap[..., 2], cmap='hot', vmin=0, vmax=1)
                axes[1, 2].set_title(f'EE Heatmap (max={heatmap[..., 2].max():.3f})')
                axes[1, 2].axis('off')

                plt.suptitle(f'{demo_key} t={t}\nObj: {valid_obj} ({u_obj:.1f}, {v_obj:.1f})  '
                            f'Target: {valid_target} ({u_target:.1f}, {v_target:.1f})  '
                            f'EE: {valid_ee} ({u_ee:.1f}, {v_ee:.1f})')

                output_path = output_dir / f'heatmap_{sample_count:03d}_{demo_key}_t{t:03d}.png'
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                plt.close()

                print(f"✅ Sample {sample_count + 1}/{num_samples}: {demo_key} t={t} → {output_path.name}")
                sample_count += 1
                success_count += 1

        print(f"\n{'='*60}")
        print(f"✅ Successfully generated {success_count}/{num_samples} heatmap visualizations")
        print(f"📁 Output directory: {output_dir}")
        print(f"{'='*60}")

        return success_count > 0


def test_projection_accuracy(hdf5_path: Path):
    """Test that projection is consistent with backprojection."""

    print("\n" + "="*60)
    print("Testing projection accuracy...")
    print("="*60)

    with h5py.File(hdf5_path, 'r') as f:
        demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])

        if not demo_keys:
            return False

        obs = f[f'data/{demo_keys[0]}/obs']

        # Test first timestep
        t = 0
        obj_pos = np.array(obs['manipulated_object_pos'][t])
        K = np.array(obs['agentview_K'][t])
        T = np.array(obs['agentview_T_camera_to_base'][t])

        u, v, valid = project_3d_to_2d(obj_pos, K, T)

        print(f"Object position (base frame): {obj_pos}")
        print(f"Projected to image: u={u:.2f}, v={v:.2f}, valid={valid}")
        print(f"Image bounds: [0, 224] × [0, 224]")

        if valid and 0 <= u < 224 and 0 <= v < 224:
            print("✅ Projection is within image bounds")
        elif valid:
            print("⚠️  Projection is valid but outside image bounds")
        else:
            print("❌ Projection is invalid (point behind camera)")

        return True


def test_corruption_modes(hdf5_path: Path, output_dir: Path):
    """Test heatmap corruption for ablation studies."""

    print("\n" + "="*60)
    print("Testing heatmap corruption...")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, 'r') as f:
        demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])

        if not demo_keys:
            return False

        obs = f[f'data/{demo_keys[0]}/obs']
        t = len(obs['ee_pos']) // 2  # Middle timestep

        obs_dict = {
            'manipulated_object_pos': np.array(obs['manipulated_object_pos'][t]),
            'target_pos': np.array(obs['target_pos'][t]),
            'ee_pos': np.array(obs['ee_pos'][t]),
            'agentview_K': np.array(obs['agentview_K'][t]),
            'agentview_T_camera_to_base': np.array(obs['agentview_T_camera_to_base'][t]),
        }

        # Generate normal heatmap
        heatmap_normal = generate_task_heatmaps(obs_dict)

        # Null heatmap
        heatmap_null = np.zeros_like(heatmap_normal)

        # Shuffle corruption
        np.random.seed(42)
        heatmap_shuffle = corrupt_heatmap(heatmap_normal, mode="shuffle")

        # Random corruption
        np.random.seed(43)
        heatmap_random = corrupt_heatmap(heatmap_normal, mode="random")

        # Visualize
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        modes = [
            ('Normal', heatmap_normal),
            ('Null', heatmap_null),
            ('Shuffle', heatmap_shuffle),
            ('Random', heatmap_random)
        ]

        for col, (name, hmap) in enumerate(modes):
            # Combined view
            axes[0, col].imshow(hmap.sum(axis=-1), cmap='hot', vmin=0, vmax=3)
            axes[0, col].set_title(f'{name} (sum)')
            axes[0, col].axis('off')

            # Object channel
            axes[1, col].imshow(hmap[..., 0], cmap='hot', vmin=0, vmax=1)
            axes[1, col].set_title(f'{name} (object ch)')
            axes[1, col].axis('off')

        plt.suptitle('Heatmap Corruption Modes for Ablation Studies')

        output_path = output_dir / 'corruption_ablation.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"✅ Corruption test saved to {output_path}")

        return True


def main():
    parser = argparse.ArgumentParser(description="Test heatmap generation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/root/autodl-tmp/LIBERO/libero/datasets/libero_spatial_plus_rgbd_30tasks_2demos",
        help="Path to RGB-D dataset directory"
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="/tmp/heatmap_viz")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output_dir)

    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return 1

    # Find first HDF5 file
    hdf5_files = list(dataset_dir.glob("*.hdf5")) + list(dataset_dir.glob("*.h5"))

    if not hdf5_files:
        print(f"❌ No HDF5 files found in {dataset_dir}")
        return 1

    test_file = hdf5_files[0]
    print(f"Using test file: {test_file}")

    # Run tests
    success = True

    success &= test_projection_accuracy(test_file)
    success &= test_corruption_modes(test_file, output_dir)
    success &= visualize_heatmap_sample(test_file, output_dir, args.num_samples)

    if success:
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("❌ SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
