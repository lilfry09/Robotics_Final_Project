"""
Diagnose: Are depth tokens from normal vs null depth actually different?
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "vla-scripts")

from train_quick_heatmap import QuickHeatmapDataset, collate_fn
from torch.utils.data import DataLoader
from prismatic.models.depth_encoder import LightweightDepthTokenEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load depth encoder
depth_encoder = LightweightDepthTokenEncoder(llm_dim=1024, grid_size=8, hidden_dim=512).to(device)

# Try to load pretrained weights if available
try:
    pretrained = torch.load('/root/autodl-tmp/openvla-oft/runs_depthvla_action_summary_v1/openvla-7b+libero_spatial_rgbd_5tasks_20demos+depth-g4+action-summary+frozen-rgb+gate-0.001+aux-0.0+b1+lr-0.0001+lora-r4+dropout-0.0/depth_encoder--latest_checkpoint.pt', map_location=device)
    depth_encoder.load_state_dict(pretrained)
    print("✓ Loaded pretrained depth encoder")
except:
    print("Using random init depth encoder")

depth_encoder.eval()

# Load data
dataset = QuickHeatmapDataset('/root/autodl-tmp/LIBERO/libero/datasets/libero_spatial_plus_rgbd_30tasks_2demos', max_episodes=2)
loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

batch = next(iter(loader))

# Normal depth
agentview_depth = batch['agentview_depth'].to(device)
eye_in_hand_depth = batch['eye_in_hand_depth'].to(device)
agentview_K = batch['agentview_K'].to(device)
agentview_T = batch['agentview_T'].to(device)
eye_in_hand_K = batch['eye_in_hand_K'].to(device)
eye_in_hand_T = batch['eye_in_hand_T'].to(device)

# Reshape for encoder
depth_values_normal = torch.cat([
    agentview_depth.squeeze(-1).unsqueeze(1),
    eye_in_hand_depth.squeeze(-1).unsqueeze(1)
], dim=1)
depth_K = torch.stack([agentview_K, eye_in_hand_K], dim=1)
depth_T = torch.stack([agentview_T, eye_in_hand_T], dim=1)

with torch.no_grad():
    tokens_normal = depth_encoder(depth_values_normal, depth_K, depth_T)

    # Null depth
    depth_values_null = torch.zeros_like(depth_values_normal)
    tokens_null = depth_encoder(depth_values_null, depth_K, depth_T)

    # Shuffle depth
    depth_values_shuffle = depth_values_normal.clone()
    for b in range(depth_values_shuffle.shape[0]):
        for v in range(depth_values_shuffle.shape[1]):
            flat = depth_values_shuffle[b, v].flatten()
            indices = torch.randperm(len(flat))
            depth_values_shuffle[b, v] = flat[indices].reshape(depth_values_shuffle.shape[2:])
    tokens_shuffle = depth_encoder(depth_values_shuffle, depth_K, depth_T)

print(f"\nDepth token statistics:")
print(f"Normal:  mean={tokens_normal.mean():.4f}, std={tokens_normal.std():.4f}, norm={tokens_normal.norm():.2f}")
print(f"Null:    mean={tokens_null.mean():.4f}, std={tokens_null.std():.4f}, norm={tokens_null.norm():.2f}")
print(f"Shuffle: mean={tokens_shuffle.mean():.4f}, std={tokens_shuffle.std():.4f}, norm={tokens_shuffle.norm():.2f}")

# Key test: L2 distance between normal and null
dist_normal_null = (tokens_normal - tokens_null).norm(dim=-1).mean()
dist_normal_shuffle = (tokens_normal - tokens_shuffle).norm(dim=-1).mean()
dist_null_shuffle = (tokens_null - tokens_shuffle).norm(dim=-1).mean()

print(f"\nL2 distances (averaged over tokens):")
print(f"Normal vs Null:    {dist_normal_null:.4f}")
print(f"Normal vs Shuffle: {dist_normal_shuffle:.4f}")
print(f"Null vs Shuffle:   {dist_null_shuffle:.4f}")

# Cosine similarity
cos_normal_null = torch.cosine_similarity(
    tokens_normal.flatten(end_dim=1),
    tokens_null.flatten(end_dim=1),
    dim=-1
).mean()

print(f"\nCosine similarity (Normal vs Null): {cos_normal_null:.4f}")

if dist_normal_null < 0.1:
    print("\n❌ PROBLEM: Normal and Null tokens are TOO SIMILAR!")
    print("   The depth encoder is not sensitive to depth content.")
elif cos_normal_null > 0.95:
    print("\n❌ PROBLEM: Normal and Null tokens are highly correlated!")
else:
    print("\n✓ Normal and Null tokens are sufficiently different")
