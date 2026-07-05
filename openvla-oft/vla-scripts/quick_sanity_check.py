"""
Sanity check: Can ANY model learn to predict heatmaps from GT heatmaps?
This tests if the data/loss/metrics are correct.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

sys.path.insert(0, "vla-scripts"); from train_quick_heatmap import QuickHeatmapDataset, collate_fn
from torch.utils.data import DataLoader
from prismatic.models.heatmap_generator import compute_heatmap_mse_loss, compute_heatmap_metrics

# Sanity check: Identity mapping (GT in → GT out)
class IdentityTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, gt_heatmap):
        return self.alpha * gt_heatmap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load data
dataset = QuickHeatmapDataset('/root/autodl-tmp/LIBERO/libero/datasets/libero_spatial_plus_rgbd_30tasks_2demos', max_episodes=12)
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0)

print(f"Total samples: {len(dataset)}")

# Test 1: Check GT heatmap statistics
print("\n[Test 1] GT Heatmap Statistics")
batch = next(iter(loader))
obj_gt = batch['object_heatmap_gt']
tgt_gt = batch['target_heatmap_gt']

print(f"Object GT: shape={obj_gt.shape}, min={obj_gt.min():.4f}, max={obj_gt.max():.4f}, mean={obj_gt.mean():.4f}")
print(f"Target GT: shape={tgt_gt.shape}, min={tgt_gt.min():.4f}, max={tgt_gt.max():.4f}, mean={tgt_gt.mean():.4f}")

# Check peak locations
for i in range(min(3, obj_gt.shape[0])):
    obj_hm = obj_gt[i, 0]
    peak_y, peak_x = torch.where(obj_hm == obj_hm.max())
    print(f"  Sample {i}: Object peak at ({peak_x[0].item()}, {peak_y[0].item()}), value={obj_hm.max():.4f}")

# Test 2: Can we learn identity mapping?
print("\n[Test 2] Identity Mapping Test")
model = IdentityTest().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for step in range(100):
    batch = next(iter(loader))
    obj_gt = batch['object_heatmap_gt'].to(device)
    obj_valid = batch['object_valid'].to(device)

    # Predict = GT (sanity)
    obj_pred = model(obj_gt)

    loss = compute_heatmap_mse_loss(obj_pred.squeeze(1), obj_gt.squeeze(1), obj_valid)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        metrics = compute_heatmap_metrics(obj_pred.squeeze(1), obj_gt.squeeze(1), obj_valid)
        print(f"Step {step}: loss={loss.item():.6f}, PSNR={metrics['psnr']:.2f} dB, alpha={model.alpha.item():.4f}")

print("\n✓ Sanity check complete")
