import unittest
import importlib.util
from pathlib import Path

import torch

_DEPTH_ENCODER_PATH = Path(__file__).resolve().parents[1] / "prismatic" / "models" / "depth_encoder.py"
_SPEC = importlib.util.spec_from_file_location("depth_encoder", _DEPTH_ENCODER_PATH)
_DEPTH_ENCODER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DEPTH_ENCODER)
LightweightDepthTokenEncoder = _DEPTH_ENCODER.LightweightDepthTokenEncoder


class DepthEncoderTest(unittest.TestCase):
    def test_depth_encoder_shapes_and_invalid_depth(self):
        encoder = LightweightDepthTokenEncoder(llm_dim=32, hidden_dim=16, grid_size=4, num_views=2)
        depth = torch.ones(2, 2, 8, 8)
        depth[0, 0, 0, 0] = float("nan")
        depth[0, 1, 0, 1] = 100.0
        intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(2, 2, 1, 1)
        intrinsics[:, :, 0, 0] = 100.0
        intrinsics[:, :, 1, 1] = 100.0
        intrinsics[:, :, 0, 2] = 4.0
        intrinsics[:, :, 1, 2] = 4.0
        extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(2, 2, 1, 1)

        tokens = encoder(depth, intrinsics, extrinsics)

        self.assertEqual(tokens.shape, (2, 2 * 4 * 4, 32))
        self.assertFalse(torch.isnan(tokens).any())

    def test_depth_encoder_accepts_channel_last_depth(self):
        encoder = LightweightDepthTokenEncoder(llm_dim=16, hidden_dim=16, grid_size=4, num_views=2)
        depth = torch.ones(1, 2, 8, 8, 1)
        intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
        extrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
        valid = torch.ones_like(depth, dtype=torch.bool)

        tokens = encoder(depth, intrinsics, extrinsics, valid)

        self.assertEqual(tokens.shape, (1, 2 * 4 * 4, 16))

    def test_depth_grid_flip_matches_libero_rgb_rotation(self):
        encoder = LightweightDepthTokenEncoder(llm_dim=8, hidden_dim=8, grid_size=2, num_views=1)
        depth = torch.ones(1, 1, 4, 4)
        intrinsics = torch.eye(3).view(1, 1, 3, 3)
        intrinsics[:, :, 0, 0] = 1.0
        intrinsics[:, :, 1, 1] = 1.0
        extrinsics = torch.eye(4).view(1, 1, 4, 4)
        valid = torch.ones_like(depth, dtype=torch.bool)

        features = torch.zeros(1, 1, 4, 4, 8)
        features[..., 5] = torch.arange(4).view(1, 1, 1, 4) / 3.0
        pooled = encoder._valid_average_pool(features, valid)
        flipped = torch.flip(pooled, dims=[2, 3])

        self.assertGreater(flipped[0, 0, 0, 0, 5], flipped[0, 0, 0, 1, 5])


if __name__ == "__main__":
    unittest.main()
