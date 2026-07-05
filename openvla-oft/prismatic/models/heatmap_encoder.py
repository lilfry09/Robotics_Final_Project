"""
Heatmap Encoder for BridgeVLA-style feature-level fusion.

Lightweight CNN that extracts spatial features from task heatmaps and fuses
them into the action head via additive residual connection.
"""

import torch
import torch.nn as nn


class HeatmapEncoder(nn.Module):
    """
    Lightweight CNN encoder for task heatmaps.

    Takes 3-channel heatmap (object, target, EE) and produces a feature vector
    that can be added to the VLA's action hidden state.
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 4096,
        alpha_init: float = 0.1,
        freeze_alpha: bool = False,
    ):
        """
        Args:
            input_channels: Number of heatmap channels (default 3: object, target, EE)
            hidden_dim: Output dimension to match action hidden state
            alpha_init: Initial fusion weight
            freeze_alpha: Whether to freeze alpha (no learning)
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim

        # Lightweight CNN feature extractor
        # Input: (B, 3, 224, 224)
        # Output: (B, hidden_dim)
        self.conv_stack = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            # 112 -> 56
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            # 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            # 28 -> 7
            nn.Conv2d(128, 256, kernel_size=3, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Project to action hidden dimension
        self.projection = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Learnable fusion weight (initialized small to not disrupt RGB baseline)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        if freeze_alpha:
            self.alpha.requires_grad = False

        # Initialize weights conservatively
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values to avoid disrupting RGB baseline."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Small initialization for projection layer
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmap: (B, C, H, W) or (B, H, W, C) heatmap tensor

        Returns:
            features: (B, hidden_dim) scaled feature vector
        """
        # Handle both (B, C, H, W) and (B, H, W, C) formats
        if heatmap.dim() == 4:
            if heatmap.shape[-1] == self.input_channels:
                # (B, H, W, C) -> (B, C, H, W)
                heatmap = heatmap.permute(0, 3, 1, 2)

        # Extract features
        features = self.conv_stack(heatmap)  # (B, 256)
        features = self.projection(features)  # (B, hidden_dim)

        # Scale by alpha
        features = features * self.alpha

        return features

    def get_alpha(self) -> float:
        """Return current fusion weight."""
        return self.alpha.item()


class HeatmapFusionWrapper(nn.Module):
    """
    Wrapper that adds heatmap fusion to an existing action head.

    This allows non-invasive integration: the original action head remains
    unchanged, and heatmap features are added as a residual.
    """

    def __init__(
        self,
        action_head: nn.Module,
        heatmap_encoder: HeatmapEncoder,
        fusion_location: str = "before_final",  # where to inject heatmap features
    ):
        """
        Args:
            action_head: Original action prediction head
            heatmap_encoder: Heatmap feature encoder
            fusion_location: Where to fuse ('before_final' or 'after_hidden')
        """
        super().__init__()

        self.action_head = action_head
        self.heatmap_encoder = heatmap_encoder
        self.fusion_location = fusion_location

    def forward(
        self,
        action_hidden: torch.Tensor,
        heatmap: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            action_hidden: (B, seq_len, hidden_dim) action hidden state from VLA
            heatmap: (B, C, H, W) task heatmap, or None for ablation
            **kwargs: Additional arguments passed to action_head

        Returns:
            action_output: (B, seq_len, action_dim) predicted actions
        """
        # If heatmap is provided, add features
        if heatmap is not None:
            heatmap_features = self.heatmap_encoder(heatmap)  # (B, hidden_dim)

            # Expand to match sequence dimension
            # action_hidden: (B, seq_len, hidden_dim)
            # heatmap_features: (B, hidden_dim) -> (B, 1, hidden_dim)
            heatmap_features = heatmap_features.unsqueeze(1)

            # Broadcast and add
            action_hidden = action_hidden + heatmap_features

        # Forward through original action head
        return self.action_head(action_hidden, **kwargs)


def create_heatmap_fusion_components(
    action_hidden_dim: int = 4096,
    heatmap_alpha_init: float = 0.1,
    freeze_alpha: bool = False,
) -> HeatmapEncoder:
    """
    Factory function to create heatmap encoder.

    Args:
        action_hidden_dim: Dimension of VLA action hidden state
        heatmap_alpha_init: Initial fusion weight
        freeze_alpha: Whether to freeze alpha

    Returns:
        heatmap_encoder: Initialized HeatmapEncoder
    """
    encoder = HeatmapEncoder(
        input_channels=3,
        hidden_dim=action_hidden_dim,
        alpha_init=heatmap_alpha_init,
        freeze_alpha=freeze_alpha,
    )

    return encoder
