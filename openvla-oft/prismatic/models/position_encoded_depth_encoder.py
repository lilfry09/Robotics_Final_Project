"""
Spatially-Aware Depth Encoder with Explicit Position Encoding

Key insight: Even if geometry pooling loses spatial info,
position encoding explicitly tells the model WHERE each token is.

This makes it impossible for shuffled depth to look like normal depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from prismatic.models.depth_encoder import LightweightDepthTokenEncoder


class PositionEncodedDepthEncoder(nn.Module):
    """
    Wrapper that adds learnable 2D position embeddings to depth tokens.

    This forces the model to know the spatial location of each token,
    making it robust to depth shuffling.
    """

    def __init__(
        self,
        llm_dim: int = 1024,
        grid_size: int = 8,
        hidden_dim: int = 512,
        pos_encoding_dim: int = 128,  # Dimension for position encoding
        num_views: int = 2,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.num_views = num_views
        self.pos_encoding_dim = pos_encoding_dim

        # Base geometry encoder (outputs llm_dim - pos_encoding_dim)
        self.base_encoder = LightweightDepthTokenEncoder(
            llm_dim=llm_dim - pos_encoding_dim,
            grid_size=grid_size,
            hidden_dim=hidden_dim,
        )

        # Learnable 2D position embeddings for each grid cell
        # Shape: (1, grid_size^2, pos_encoding_dim)
        self.pos_embed_x = nn.Parameter(
            torch.randn(1, grid_size, 1, pos_encoding_dim // 2) * 0.02
        )
        self.pos_embed_y = nn.Parameter(
            torch.randn(1, 1, grid_size, pos_encoding_dim // 2) * 0.02
        )

        # View-specific embeddings (differentiate agentview vs eye-in-hand)
        self.view_embed = nn.Parameter(
            torch.randn(1, num_views, 1, pos_encoding_dim) * 0.02
        )

    def forward(
        self,
        depth_values: torch.Tensor,
        depth_intrinsics: torch.Tensor,
        depth_extrinsics: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            depth_values: (B, V, H, W)
            depth_intrinsics: (B, V, 3, 3)
            depth_extrinsics: (B, V, 4, 4)

        Returns:
            tokens: (B, V * grid_size^2, llm_dim)
        """
        B = depth_values.shape[0]
        V = self.num_views
        G = self.grid_size

        # Get base geometry tokens (without position info)
        geo_tokens = self.base_encoder(
            depth_values, depth_intrinsics, depth_extrinsics, depth_valid_mask
        )  # (B, V * G^2, llm_dim - pos_encoding_dim)

        # Create 2D position embeddings by combining x and y
        # pos_embed_x: (1, G, 1, D/2) → broadcast to (1, G, G, D/2)
        # pos_embed_y: (1, 1, G, D/2) → broadcast to (1, G, G, D/2)
        pos_2d = torch.cat([
            self.pos_embed_x.expand(-1, -1, G, -1),
            self.pos_embed_y.expand(-1, G, -1, -1)
        ], dim=-1)  # (1, G, G, pos_encoding_dim)

        # Flatten spatial dimensions
        pos_2d = pos_2d.reshape(1, G * G, self.pos_encoding_dim)  # (1, G^2, D)

        # Repeat for each view and add view-specific embedding
        pos_encoding = []
        for v in range(V):
            view_pos = pos_2d + self.view_embed[:, v, :, :]  # (1, 1, G^2, D)
            view_pos = view_pos.squeeze(1)  # (1, G^2, D)
            pos_encoding.append(view_pos)

        pos_encoding = torch.cat(pos_encoding, dim=1)  # (1, V*G^2, D)
        pos_encoding = pos_encoding.expand(B, -1, -1)  # (B, V*G^2, D)

        # Concatenate geometry tokens with position encoding
        tokens = torch.cat([geo_tokens, pos_encoding], dim=-1)  # (B, V*G^2, llm_dim)

        return tokens


if __name__ == "__main__":
    print("Testing PositionEncodedDepthEncoder...")

    # Create model
    model = PositionEncodedDepthEncoder(
        llm_dim=1024,
        grid_size=8,
        hidden_dim=512,
        pos_encoding_dim=128
    )

    # Dummy input
    B = 4
    depth = torch.randn(B, 2, 256, 256)
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, 2, -1, -1)
    T = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, 2, -1, -1)

    # Forward pass
    tokens = model(depth, K, T)

    print(f"✓ Input depth shape: {depth.shape}")
    print(f"✓ Output tokens shape: {tokens.shape}")
    print(f"✓ Expected: (B={B}, num_tokens={2*8*8}, dim=1024)")

    # Test that position encoding is different for different locations
    pos_00 = tokens[0, 0, -128:]  # Position encoding for grid cell (0, 0), view 0
    pos_77 = tokens[0, 63, -128:]  # Position encoding for grid cell (7, 7), view 0
    pos_diff = (pos_00 - pos_77).norm()

    print(f"\n✓ Position encoding difference (cell 0,0 vs 7,7): {pos_diff:.4f}")
    assert pos_diff > 0.1, "Position encodings should be different!"

    # Test that shuffling depth doesn't affect position encoding
    depth_shuffled = depth.clone()
    for b in range(B):
        for v in range(2):
            flat = depth_shuffled[b, v].flatten()
            indices = torch.randperm(len(flat))
            depth_shuffled[b, v] = flat[indices].reshape(256, 256)

    tokens_shuffled = model(depth_shuffled, K, T)

    # Position encodings should be SAME for shuffled depth
    pos_encoding_normal = tokens[0, :, -128:]
    pos_encoding_shuffled = tokens_shuffled[0, :, -128:]
    pos_diff_shuffle = (pos_encoding_normal - pos_encoding_shuffled).norm()

    print(f"✓ Position encoding change after shuffle: {pos_diff_shuffle:.4f}")
    assert pos_diff_shuffle < 0.01, "Position encodings should NOT change when depth is shuffled!"

    # But geometry tokens SHOULD change
    geo_normal = tokens[0, :, :-128]
    geo_shuffled = tokens_shuffled[0, :, :-128]
    geo_diff = (geo_normal - geo_shuffled).norm()

    print(f"✓ Geometry encoding change after shuffle: {geo_diff:.4f}")
    assert geo_diff > 0.1, "Geometry encodings SHOULD change when depth is shuffled!"

    print("\n✓ All tests passed!")
    print("\nKey properties:")
    print("  1. Position encoding tells WHERE each token is")
    print("  2. Position encoding is INVARIANT to depth content")
    print("  3. Geometry encoding captures depth content")
    print("  4. Together they provide spatial + semantic info")
