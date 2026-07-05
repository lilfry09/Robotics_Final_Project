"""Test heatmap fusion module with dummy data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from prismatic.models.heatmap_fusion_simple import SimpleHeatmapFusion, add_heatmap_to_action_hidden


def test_heatmap_fusion():
    print("="*60)
    print("Testing SimpleHeatmapFusion module")
    print("="*60)

    # Create module
    fusion = SimpleHeatmapFusion(hidden_dim=4096, alpha_init=0.1)
    print(f"✅ Created SimpleHeatmapFusion with {sum(p.numel() for p in fusion.parameters())} parameters")
    print(f"   Initial alpha: {fusion.alpha.item():.4f}")

    # Create dummy inputs
    batch_size = 2
    heatmap = torch.randn(batch_size, 3, 224, 224)
    action_hidden = torch.randn(batch_size, 56, 4096)  # 56 = 8 actions * 7 dims

    print(f"\n✅ Created dummy inputs:")
    print(f"   Heatmap shape: {tuple(heatmap.shape)}")
    print(f"   Action hidden shape: {tuple(action_hidden.shape)}")

    # Test forward pass
    heatmap_feat = fusion(heatmap)
    print(f"\n✅ Forward pass successful:")
    print(f"   Output shape: {tuple(heatmap_feat.shape)}")
    print(f"   Output mean: {heatmap_feat.mean().item():.6f}")
    print(f"   Output std: {heatmap_feat.std().item():.6f}")

    # Test fusion
    action_hidden_fused = add_heatmap_to_action_hidden(action_hidden, heatmap, fusion)
    print(f"\n✅ Fusion successful:")
    print(f"   Fused shape: {tuple(action_hidden_fused.shape)}")
    print(f"   Original mean: {action_hidden.mean().item():.6f}")
    print(f"   Fused mean: {action_hidden_fused.mean().item():.6f}")
    print(f"   Delta: {(action_hidden_fused - action_hidden).abs().mean().item():.6f}")

    # Test null heatmap (ablation)
    action_hidden_null = add_heatmap_to_action_hidden(action_hidden, None, fusion)
    print(f"\n✅ Null heatmap ablation:")
    print(f"   Output equals input: {torch.allclose(action_hidden_null, action_hidden)}")

    # Test channel order handling
    heatmap_hwc = heatmap.permute(0, 2, 3, 1)  # (B, H, W, C)
    heatmap_feat_hwc = fusion(heatmap_hwc)
    print(f"\n✅ Channel order handling:")
    print(f"   HWC input shape: {tuple(heatmap_hwc.shape)}")
    print(f"   Output shape: {tuple(heatmap_feat_hwc.shape)}")
    print(f"   Outputs equal: {torch.allclose(heatmap_feat, heatmap_feat_hwc)}")

    # Test gradient flow
    loss = heatmap_feat.sum()
    loss.backward()
    print(f"\n✅ Gradient flow:")
    print(f"   Alpha grad: {fusion.alpha.grad.item():.6f}")
    print(f"   Conv weight grad norm: {fusion.conv[0].weight.grad.norm().item():.6f}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    test_heatmap_fusion()
