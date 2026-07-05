"""
8-Hour Quick Validation: Train standalone heatmap predictor

Goal: Prove that depth information CAN be causally used for spatial prediction.

Strategy:
1. Train: depth features → heatmap prediction
2. Evaluate: normal depth vs null depth vs shuffled depth
3. Gate: If normal >> null, depth is being used causally

This validates the approach WITHOUT needing full VLA training + 30-task rollout.
"""

import sys
import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prismatic.models.heatmap_generator import (
    create_heatmap_labels_batch,
    compute_heatmap_mse_loss,
    compute_heatmap_metrics,
)
from prismatic.models.minimal_heatmap_head import MinimalHeatmapHead
from prismatic.models.position_encoded_depth_encoder import PositionEncodedDepthEncoder


class QuickHeatmapDataset(Dataset):
    """Load depth + GT heatmaps from LIBERO-Plus symbolic data."""

    def __init__(self, hdf5_dir, max_episodes=None):
        self.samples = []

        hdf5_files = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
        if max_episodes:
            hdf5_files = hdf5_files[:max_episodes]

        print(f"Loading data from {len(hdf5_files)} episodes...")

        for hdf5_path in tqdm(hdf5_files):
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    # Try demo_0, if not exist, use first available demo
                    if 'data/demo_0' in f:
                        demo = f['data/demo_0']
                    else:
                        demo_keys = list(f['data'].keys())
                        if not demo_keys:
                            continue
                        demo = f[f'data/{demo_keys[0]}']

                    T = len(demo['obs']['agentview_rgb'])

                    # Load all timesteps
                    for t in range(T):
                        self.samples.append({
                            'hdf5_path': hdf5_path,
                            'timestep': t,
                        })
            except Exception as e:
                print(f"Warning: Skipping {hdf5_path}: {e}")
                continue

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with h5py.File(sample['hdf5_path'], 'r') as f:
            # Find first available demo
            if 'data/demo_0' in f:
                demo = f['data/demo_0']
            else:
                demo_keys = list(f['data'].keys())
                demo = f[f'data/{demo_keys[0]}']

            t = sample['timestep']

            # Load data
            data = {
                'agentview_depth_m': demo['obs']['agentview_depth_m'][t],  # (256, 256, 1)
                'eye_in_hand_depth_m': demo['obs']['eye_in_hand_depth_m'][t],
                'agentview_K': demo['obs']['agentview_K'][t],  # (3, 3)
                'agentview_T_camera_to_base': demo['obs']['agentview_T_camera_to_base'][t],  # (4, 4)
                'eye_in_hand_K': demo['obs']['eye_in_hand_K'][t],
                'eye_in_hand_T_camera_to_base': demo['obs']['eye_in_hand_T_camera_to_base'][t],
                'manipulated_object_pos': demo['obs']['manipulated_object_pos'][t],  # (3,)
                'target_pos': demo['obs']['target_pos'][t],  # (3,)
            }

        return data


def collate_fn(batch):
    """Collate batch and generate GT heatmaps."""
    B = len(batch)

    # Stack depth maps
    agentview_depth = torch.stack([
        torch.from_numpy(item['agentview_depth_m']).float() for item in batch
    ])  # (B, 256, 256, 1)

    eye_in_hand_depth = torch.stack([
        torch.from_numpy(item['eye_in_hand_depth_m']).float() for item in batch
    ])  # (B, 256, 256, 1)

    # Stack camera params
    agentview_K = torch.stack([torch.from_numpy(item['agentview_K']).float() for item in batch])
    agentview_T = torch.stack([torch.from_numpy(item['agentview_T_camera_to_base']).float() for item in batch])
    eye_in_hand_K = torch.stack([torch.from_numpy(item['eye_in_hand_K']).float() for item in batch])
    eye_in_hand_T = torch.stack([torch.from_numpy(item['eye_in_hand_T_camera_to_base']).float() for item in batch])

    # Stack object/target positions
    object_pos = torch.stack([torch.from_numpy(item['manipulated_object_pos']).float() for item in batch])
    target_pos = torch.stack([torch.from_numpy(item['target_pos']).float() for item in batch])

    # Generate GT heatmaps (use agentview for now)
    heatmap_dict = create_heatmap_labels_batch(
        object_pos,
        target_pos,
        agentview_K,
        agentview_T,
        image_height=256,
        image_width=256,
        sigma=15.0
    )

    return {
        'agentview_depth': agentview_depth,
        'eye_in_hand_depth': eye_in_hand_depth,
        'agentview_K': agentview_K,
        'agentview_T': agentview_T,
        'eye_in_hand_K': eye_in_hand_K,
        'eye_in_hand_T': eye_in_hand_T,
        'object_heatmap_gt': heatmap_dict['object_heatmap'].unsqueeze(1),  # (B, 1, 256, 256)
        'target_heatmap_gt': heatmap_dict['target_heatmap'].unsqueeze(1),
        'object_valid': heatmap_dict['object_valid'],
        'target_valid': heatmap_dict['target_valid'],
    }


def train_quick(args):
    """Quick training loop."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    dataset = QuickHeatmapDataset(args.data_dir, max_episodes=args.max_episodes)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Models - Position-Encoded Depth Encoder (grid_size=8 with explicit position info)
    depth_encoder = PositionEncodedDepthEncoder(
        llm_dim=1024,
        grid_size=8,
        hidden_dim=512,
        pos_encoding_dim=128,  # 128 dims for spatial position
        num_views=2
    ).to(device)

    heatmap_head = MinimalHeatmapHead(
        depth_token_dim=1024,
        num_depth_tokens=128,  # 2 views × 8×8 = 128 (was 32)
        hidden_dim=256
    ).to(device)

    # Optimizer - INCREASED learning rate
    params = list(depth_encoder.parameters()) + list(heatmap_head.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    # Training loop
    print(f"\nTraining for {args.max_steps} steps...")

    global_step = 0
    best_test_psnr = 0

    for epoch in range(100):  # Max epochs
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            # Move to device
            agentview_depth = batch['agentview_depth'].to(device)
            eye_in_hand_depth = batch['eye_in_hand_depth'].to(device)
            agentview_K = batch['agentview_K'].to(device)
            agentview_T = batch['agentview_T'].to(device)
            eye_in_hand_K = batch['eye_in_hand_K'].to(device)
            eye_in_hand_T = batch['eye_in_hand_T'].to(device)

            object_hm_gt = batch['object_heatmap_gt'].to(device)
            target_hm_gt = batch['target_heatmap_gt'].to(device)
            object_valid = batch['object_valid'].to(device)
            target_valid = batch['target_valid'].to(device)

            # Encode depth - reshape to match encoder expected format
            # Encoder expects: (B, V, H, W), (B, V, 3, 3), (B, V, 4, 4)
            B = agentview_depth.shape[0]

            # Stack views: (B, 2, H, W, 1) -> (B, 2, H, W)
            depth_values = torch.cat([
                agentview_depth.squeeze(-1).unsqueeze(1),  # (B, 1, H, W)
                eye_in_hand_depth.squeeze(-1).unsqueeze(1)  # (B, 1, H, W)
            ], dim=1)  # (B, 2, H, W)

            # Stack intrinsics and extrinsics
            depth_K = torch.stack([agentview_K, eye_in_hand_K], dim=1)  # (B, 2, 3, 3)
            depth_T = torch.stack([agentview_T, eye_in_hand_T], dim=1)  # (B, 2, 4, 4)

            # Encode depth
            depth_tokens = depth_encoder(
                depth_values, depth_K, depth_T
            )  # (B, 32, 1024)

            # Predict heatmaps
            object_hm_pred, target_hm_pred = heatmap_head(depth_tokens)

            # Compute loss
            loss_obj = compute_heatmap_mse_loss(
                object_hm_pred.squeeze(1), object_hm_gt.squeeze(1), object_valid
            )
            loss_tgt = compute_heatmap_mse_loss(
                target_hm_pred.squeeze(1), target_hm_gt.squeeze(1), target_valid
            )

            loss = loss_obj + loss_tgt

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  # Add gradient clipping
            optimizer.step()
            scheduler.step()  # Update learning rate

            global_step += 1

            if global_step % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step}/{args.max_steps}: loss={loss.item():.4f}, "
                      f"obj={loss_obj.item():.4f}, tgt={loss_tgt.item():.4f}, lr={current_lr:.6f}")

            if global_step % 200 == 0:
                # Quick eval
                test_metrics = evaluate_quick(depth_encoder, heatmap_head, test_loader, device, max_batches=10)
                print(f"  Test - Object PSNR: {test_metrics['object_psnr']:.2f} dB, "
                      f"Target PSNR: {test_metrics['target_psnr']:.2f} dB")

                # Save best
                if test_metrics['object_psnr'] > best_test_psnr:
                    best_test_psnr = test_metrics['object_psnr']
                    save_checkpoint(depth_encoder, heatmap_head, args.save_dir, 'best')

        if global_step >= args.max_steps:
            break

    # Final eval
    print("\n" + "="*70)
    print("Final Evaluation")
    print("="*70)

    test_metrics = evaluate_quick(depth_encoder, heatmap_head, test_loader, device)
    print(f"Object PSNR: {test_metrics['object_psnr']:.2f} dB")
    print(f"Target PSNR: {test_metrics['target_psnr']:.2f} dB")
    print(f"Object peak distance: {test_metrics['object_peak_dist']:.2f} pixels")
    print(f"Target peak distance: {test_metrics['target_peak_dist']:.2f} pixels")

    # Save final
    save_checkpoint(depth_encoder, heatmap_head, args.save_dir, 'final')

    return test_metrics


def evaluate_quick(depth_encoder, heatmap_head, test_loader, device, max_batches=None):
    """Quick evaluation."""
    depth_encoder.eval()
    heatmap_head.eval()

    all_obj_metrics = []
    all_tgt_metrics = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if max_batches and i >= max_batches:
                break

            agentview_depth = batch['agentview_depth'].to(device)
            eye_in_hand_depth = batch['eye_in_hand_depth'].to(device)
            agentview_K = batch['agentview_K'].to(device)
            agentview_T = batch['agentview_T'].to(device)
            eye_in_hand_K = batch['eye_in_hand_K'].to(device)
            eye_in_hand_T = batch['eye_in_hand_T'].to(device)

            object_hm_gt = batch['object_heatmap_gt'].to(device)
            target_hm_gt = batch['target_heatmap_gt'].to(device)
            object_valid = batch['object_valid'].to(device)
            target_valid = batch['target_valid'].to(device)

            # Encode + predict
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

    depth_encoder.train()
    heatmap_head.train()

    # Aggregate
    return {
        'object_psnr': np.mean([m['psnr'] for m in all_obj_metrics]),
        'target_psnr': np.mean([m['psnr'] for m in all_tgt_metrics]),
        'object_peak_dist': np.mean([m['peak_distance'] for m in all_obj_metrics]),
        'target_peak_dist': np.mean([m['peak_distance'] for m in all_tgt_metrics]),
    }


def save_checkpoint(depth_encoder, heatmap_head, save_dir, name):
    """Save checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'depth_encoder': depth_encoder.state_dict(),
        'heatmap_head': heatmap_head.state_dict(),
    }, os.path.join(save_dir, f'checkpoint_{name}.pt'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/root/autodl-tmp/LIBERO/libero/datasets/libero_spatial_plus_rgbd_30tasks_2demos')
    parser.add_argument('--save_dir', type=str,
                        default='/root/autodl-tmp/openvla-oft/runs_quick_heatmap')
    parser.add_argument('--max_episodes', type=int, default=None, help='Max episodes to load')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_steps', type=int, default=500)
    args = parser.parse_args()

    train_quick(args)
