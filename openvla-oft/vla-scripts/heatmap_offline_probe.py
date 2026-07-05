"""
Offline heatmap probe: verify that heatmaps produce action prediction differences.

This is a GO/NO-GO gate before training. We train a simple MLP to predict actions
from heatmaps. If normal heatmap doesn't beat null/corrupt by a significant margin,
don't proceed to VLA training.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from prismatic.models.heatmap_generator import generate_task_heatmaps, corrupt_heatmap


class SimpleHeatmapActionPredictor(nn.Module):
    """Simple MLP that predicts actions from heatmaps."""

    def __init__(self, action_dim=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, heatmap):
        if heatmap.shape[-1] == 3:
            heatmap = heatmap.permute(0, 3, 1, 2)
        feat = self.conv(heatmap)
        return self.mlp(feat)


def load_heatmap_action_pairs(dataset_dir: Path, num_samples: int = 500):
    """Load heatmaps and corresponding actions from dataset."""

    heatmaps_normal = []
    heatmaps_null = []
    heatmaps_corrupt = []
    actions = []

    hdf5_files = list(dataset_dir.glob("*.hdf5")) + list(dataset_dir.glob("*.h5"))

    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files in {dataset_dir}")

    sample_count = 0

    for hdf5_path in hdf5_files:
        if sample_count >= num_samples:
            break

        with h5py.File(hdf5_path, 'r') as f:
            demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])

            for demo_key in demo_keys:
                if sample_count >= num_samples:
                    break

                obs = f[f'data/{demo_key}/obs']
                actions_demo = f[f'data/{demo_key}/actions'][()]
                num_steps = len(actions_demo)

                # Check required fields
                required = [
                    'manipulated_object_pos', 'target_pos', 'ee_pos',
                    'agentview_K', 'agentview_T_camera_to_base'
                ]
                if not all(field in obs for field in required):
                    continue

                # Sample uniformly from episode
                step_indices = np.linspace(0, num_steps - 1, min(10, num_steps), dtype=int)

                for t in step_indices:
                    if sample_count >= num_samples:
                        break

                    obs_dict = {
                        'manipulated_object_pos': np.array(obs['manipulated_object_pos'][t]),
                        'target_pos': np.array(obs['target_pos'][t]),
                        'ee_pos': np.array(obs['ee_pos'][t]),
                        'agentview_K': np.array(obs['agentview_K'][t]),
                        'agentview_T_camera_to_base': np.array(obs['agentview_T_camera_to_base'][t]),
                    }

                    # Check validity
                    if not all(np.isfinite(obs_dict[k]).all() for k in ['manipulated_object_pos', 'target_pos', 'ee_pos']):
                        continue

                    try:
                        # Generate normal heatmap
                        heatmap_normal = generate_task_heatmaps(obs_dict, image_size=(224, 224), sigma=5.0)

                        # Null heatmap
                        heatmap_null = np.zeros_like(heatmap_normal)

                        # Corrupt heatmap
                        np.random.seed(sample_count + 42)
                        heatmap_corrupt = corrupt_heatmap(heatmap_normal, mode="random")

                        # Action
                        action = actions_demo[t][:7]  # First 7 dims

                        heatmaps_normal.append(heatmap_normal)
                        heatmaps_null.append(heatmap_null)
                        heatmaps_corrupt.append(heatmap_corrupt)
                        actions.append(action)

                        sample_count += 1

                    except Exception as e:
                        continue

    print(f"Loaded {len(actions)} samples from {dataset_dir}")

    return (
        np.stack(heatmaps_normal),
        np.stack(heatmaps_null),
        np.stack(heatmaps_corrupt),
        np.stack(actions),
    )


def train_and_evaluate(heatmaps, actions, test_split=0.3, epochs=15):
    """Train predictor and return test RMSE."""

    # Split
    n = len(actions)
    n_train = int(n * (1 - test_split))

    heatmaps_train = torch.from_numpy(heatmaps[:n_train]).float()
    actions_train = torch.from_numpy(actions[:n_train]).float()
    heatmaps_test = torch.from_numpy(heatmaps[n_train:]).float()
    actions_test = torch.from_numpy(actions[n_train:]).float()

    # Create datasets
    train_dataset = TensorDataset(heatmaps_train, actions_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Model
    model = SimpleHeatmapActionPredictor(action_dim=actions.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    # Train
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for heatmap_batch, action_batch in train_loader:
            optimizer.zero_grad()
            pred = model(heatmap_batch)
            loss = criterion(pred, action_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds_test = model(heatmaps_test)
        rmse = torch.sqrt(((preds_test - actions_test) ** 2).mean()).item()
        mae = (preds_test - actions_test).abs().mean().item()

    return rmse, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="/root/autodl-tmp/LIBERO/libero/datasets/libero_spatial_plus_rgbd_30tasks_2demos",
    )
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="experiments/logs/heatmap_offline_probe.json")
    parser.add_argument("--epochs", type=int, default=15)

    args = parser.parse_args()

    print("="*60)
    print("Heatmap Offline Probe")
    print("="*60)

    # Load data
    print("\nLoading data...")
    heatmaps_normal, heatmaps_null, heatmaps_corrupt, actions = load_heatmap_action_pairs(
        Path(args.dataset), num_samples=args.num_samples
    )

    print(f"Loaded {len(actions)} samples")
    print(f"Heatmap shape: {heatmaps_normal.shape}")
    print(f"Action shape: {actions.shape}")

    # Train on normal
    print(f"\nTraining on NORMAL heatmaps ({args.epochs} epochs)...")
    rmse_normal, mae_normal = train_and_evaluate(heatmaps_normal, actions, epochs=args.epochs)

    # Evaluate on null
    print(f"\nTraining on NULL heatmaps ({args.epochs} epochs)...")
    rmse_null, mae_null = train_and_evaluate(heatmaps_null, actions, epochs=args.epochs)

    # Evaluate on corrupt
    print(f"\nTraining on CORRUPT heatmaps ({args.epochs} epochs)...")
    rmse_corrupt, mae_corrupt = train_and_evaluate(heatmaps_corrupt, actions, epochs=args.epochs)

    # Results
    results = {
        "dataset": str(args.dataset),
        "num_samples": len(actions),
        "normal_rmse": float(rmse_normal),
        "normal_mae": float(mae_normal),
        "null_rmse": float(rmse_null),
        "null_mae": float(mae_null),
        "corrupt_rmse": float(rmse_corrupt),
        "corrupt_mae": float(mae_corrupt),
        "advantage_over_null": float(rmse_null - rmse_normal),
        "advantage_over_corrupt": float(rmse_corrupt - rmse_normal),
    }

    # Print summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Normal  RMSE: {rmse_normal:.4f}  MAE: {mae_normal:.4f}")
    print(f"Null    RMSE: {rmse_null:.4f}  MAE: {mae_null:.4f}")
    print(f"Corrupt RMSE: {rmse_corrupt:.4f}  MAE: {mae_corrupt:.4f}")
    print(f"\nAdvantage over null:    {rmse_null - rmse_normal:+.4f}")
    print(f"Advantage over corrupt: {rmse_corrupt - rmse_normal:+.4f}")

    # GO/NO-GO decision
    print("\n" + "="*60)
    print("GO/NO-GO DECISION")
    print("="*60)

    threshold = 0.10
    if (rmse_null - rmse_normal) >= threshold and (rmse_corrupt - rmse_normal) >= threshold:
        print(f"✅ GO: Normal beats Null by {rmse_null - rmse_normal:.4f} (>= {threshold})")
        print(f"✅ GO: Normal beats Corrupt by {rmse_corrupt - rmse_normal:.4f} (>= {threshold})")
        print("\n🟢 PROCEED TO TRAINING")
        results["decision"] = "GO"
    else:
        print(f"❌ NO-GO: Advantage too small (threshold: {threshold})")
        print("\n🔴 DO NOT TRAIN - Heatmap does not provide sufficient signal")
        results["decision"] = "NO-GO"

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Results saved to {output_path}")
    print("="*60)

    return 0 if results["decision"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
