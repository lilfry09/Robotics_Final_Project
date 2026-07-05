"""
Quick heatmap fusion implementation for Phase 2.

This module provides a minimal heatmap integration that:
1. Adds heatmap generation to dataset loading
2. Provides a simple additive fusion to action hidden states
3. Avoids modifying existing VLA/depth infrastructure
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

from prismatic.models.heatmap_generator import generate_task_heatmaps, corrupt_heatmap


def generate_heatmap_from_batch_sample(
    ep: Dict,
    t: int,
    image_size: tuple = (224, 224),
    sigma: float = 5.0,
    view: str = "agentview"
) -> Optional[np.ndarray]:
    """
    Generate heatmap for a single training sample.

    Args:
        ep: Episode dictionary from dataset
        t: Timestep
        image_size: Output heatmap size
        sigma: Gaussian sigma
        view: Camera view name

    Returns:
        heatmap: (H, W, 3) float32 array or None if fields missing
    """
    try:
        # Build obs_dict
        obs_dict = {}

        # Required fields
        required_keys = [
            'manipulated_object_pos', 'target_pos', 'ee_pos',
            f'{view}_K', f'{view}_T_camera_to_base'
        ]

        for key in required_keys:
            if key not in ep:
                return None
            obs_dict[key] = ep[key][t] if ep[key].ndim > 1 else ep[key]

        # Generate heatmap
        heatmap = generate_task_heatmaps(obs_dict, image_size=image_size, sigma=sigma, view=view)

        return heatmap

    except Exception as e:
        # Fail silently and return None if any issue
        return None


class SimpleHeatmapFusion(nn.Module):
    """
    Minimal heatmap fusion module.

    Extracts features from heatmap and adds them to action hidden state.
    """

    def __init__(self, hidden_dim: int = 4096, alpha_init: float = 0.1):
        super().__init__()

        # Simple CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),  # 224->112
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, 2),  # 112->56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 56->28
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Project to hidden dim
        self.proj = nn.Linear(128, hidden_dim)

        # Fusion weight
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Initialize conservatively
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmap: (B, 3, H, W) or (B, H, W, 3)

        Returns:
            features: (B, hidden_dim)
        """
        # Handle channel order
        if heatmap.shape[-1] == 3:
            heatmap = heatmap.permute(0, 3, 1, 2)

        # Extract features
        feat = self.conv(heatmap)  # (B, 128)
        feat = self.proj(feat)  # (B, hidden_dim)

        # Scale by alpha
        return feat * self.alpha


def add_heatmap_to_action_hidden(
    action_hidden: torch.Tensor,
    heatmap: Optional[torch.Tensor],
    heatmap_fusion: Optional[SimpleHeatmapFusion]
) -> torch.Tensor:
    """
    Add heatmap features to action hidden state.

    Args:
        action_hidden: (B, seq_len, hidden_dim)
        heatmap: (B, 3, H, W) or None
        heatmap_fusion: Fusion module or None

    Returns:
        action_hidden: (B, seq_len, hidden_dim) with heatmap added
    """
    if heatmap is None or heatmap_fusion is None:
        return action_hidden

    # Extract features
    heatmap_feat = heatmap_fusion(heatmap)  # (B, hidden_dim)

    # Broadcast and add
    heatmap_feat = heatmap_feat.unsqueeze(1)  # (B, 1, hidden_dim)
    action_hidden = action_hidden + heatmap_feat

    return action_hidden
