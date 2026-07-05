"""
Minimal Heatmap Prediction Head for 8-hour validation.

This is a simplified version focusing on PROOF OF CONCEPT:
- Can depth features predict spatial heatmaps?
- Does normal depth beat null depth?

NOT trying to achieve SOTA, just validate the approach works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalHeatmapHead(nn.Module):
    """
    Extremely simple heatmap predictor for quick validation.

    Input:  depth geometry features (B, 32, depth_dim)
    Output: object_heatmap (B, 1, H, W)
            target_heatmap (B, 1, H, W)
    """

    def __init__(
        self,
        depth_token_dim=1024,  # Will be set based on actual depth encoder
        num_depth_tokens=32,   # 2 views × 4×4 grid
        heatmap_size=256,
        hidden_dim=256,
    ):
        super().__init__()

        self.num_depth_tokens = num_depth_tokens
        self.heatmap_size = heatmap_size

        # Pool depth tokens to global context
        self.global_pool = nn.Sequential(
            nn.Linear(depth_token_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Decode to spatial features
        # Start from 4x4, upsample to 256x256
        self.spatial_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4 * 4),  # 256 * 16
            nn.ReLU(),
            nn.Unflatten(1, (hidden_dim, 4, 4)),

            # 4x4 -> 8x8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(),

            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1),
            nn.ReLU(),

            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, 3, padding=1),
            nn.ReLU(),

            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim // 8, 64, 3, padding=1),
            nn.ReLU(),

            # 64x64 -> 128x128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            # 128x128 -> 256x256
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
        )

        # Separate prediction heads
        self.object_head = nn.Conv2d(16, 1, 1)
        self.target_head = nn.Conv2d(16, 1, 1)

    def forward(self, depth_tokens):
        """
        Args:
            depth_tokens: (B, num_tokens, depth_dim) depth geometry features

        Returns:
            object_heatmap: (B, 1, H, W)
            target_heatmap: (B, 1, H, W)
        """
        B = depth_tokens.shape[0]

        # Global pooling
        pooled = self.global_pool(depth_tokens.mean(dim=1))  # (B, hidden_dim)

        # Decode to spatial
        spatial = self.spatial_decoder(pooled)  # (B, 16, 256, 256)

        # Predict heatmaps
        object_hm = torch.sigmoid(self.object_head(spatial))  # (B, 1, 256, 256)
        target_hm = torch.sigmoid(self.target_head(spatial))  # (B, 1, 256, 256)

        return object_hm, target_hm


if __name__ == "__main__":
    print("Testing MinimalHeatmapHead...")

    # Smoke test
    model = MinimalHeatmapHead(depth_token_dim=1024, hidden_dim=256)

    # Dummy input
    dummy_depth = torch.randn(4, 32, 1024)

    # Forward pass
    obj_hm, tgt_hm = model(dummy_depth)

    print(f"✓ Input shape: {dummy_depth.shape}")
    print(f"✓ Object heatmap shape: {obj_hm.shape}")
    print(f"✓ Target heatmap shape: {tgt_hm.shape}")
    print(f"✓ Object heatmap range: [{obj_hm.min():.3f}, {obj_hm.max():.3f}]")
    print(f"✓ Target heatmap range: [{tgt_hm.min():.3f}, {tgt_hm.max():.3f}]")

    # Check output is in valid range
    assert (obj_hm >= 0).all() and (obj_hm <= 1).all(), "Object heatmap not in [0,1]"
    assert (tgt_hm >= 0).all() and (tgt_hm <= 1).all(), "Target heatmap not in [0,1]"

    print("\n✓ MinimalHeatmapHead smoke test passed!")
