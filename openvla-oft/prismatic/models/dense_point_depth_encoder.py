"""Dense point-token depth encoder for the next RGB-D experiments.

This encoder keeps many sampled metric 3D points instead of first pooling depth
into a coarse spatial grid. It is meant for object/action-query fusion where the
action head attends over a set of candidate 3D locations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DensePointDepthTokenEncoder(nn.Module):
    """Encode sampled RGB-D geometry into LLM-space point tokens.

    Args:
        llm_dim: Output token dimension.
        hidden_dim: MLP hidden size.
        num_points_per_view: Number of regularly sampled depth points per view.
        depth_min_m: Minimum valid metric depth.
        depth_max_m: Maximum valid metric depth.
        num_views: Expected number of camera views.

    Input shapes match ``LightweightDepthTokenEncoder``:

    - ``depth_values``: ``(B, V, H, W)``
    - ``depth_intrinsics``: ``(B, V, 3, 3)``
    - ``depth_extrinsics``: ``(B, V, 4, 4)``

    Output:

    - point tokens: ``(B, V * num_points_per_view, llm_dim)``
    """

    def __init__(
        self,
        llm_dim: int,
        hidden_dim: int = 256,
        num_points_per_view: int = 1024,
        depth_min_m: float = 0.01,
        depth_max_m: float = 5.0,
        num_views: int = 2,
        alpha_init: float = 0.01,
    ) -> None:
        super().__init__()
        if num_points_per_view <= 0:
            raise ValueError("num_points_per_view must be positive")
        self.llm_dim = int(llm_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_points_per_view = int(num_points_per_view)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.num_views = int(num_views)
        self.depth_num_tokens = self.num_views * self.num_points_per_view
        self.encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.ablation_mode = "none"
        self.shuffle_seed = 0

    def forward(
        self,
        depth_values: torch.Tensor,
        depth_intrinsics: torch.Tensor,
        depth_extrinsics: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
        ee_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.compute_point_features(
            depth_values=depth_values,
            depth_intrinsics=depth_intrinsics,
            depth_extrinsics=depth_extrinsics,
            depth_valid_mask=depth_valid_mask,
            ee_pos=ee_pos,
        )
        features = self._apply_ablation(features)
        tokens = self.encoder(features.to(dtype=next(self.encoder.parameters()).dtype))
        return self.alpha.to(tokens.dtype) * tokens

    def forward_summary(
        self,
        depth_values: torch.Tensor,
        depth_intrinsics: torch.Tensor,
        depth_extrinsics: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
        ee_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.forward(depth_values, depth_intrinsics, depth_extrinsics, depth_valid_mask, ee_pos=ee_pos)
        return tokens.mean(dim=1)

    def compute_point_features(
        self,
        depth_values: torch.Tensor,
        depth_intrinsics: torch.Tensor,
        depth_extrinsics: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
        ee_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if depth_values.ndim == 5 and depth_values.shape[-1] == 1:
            depth_values = depth_values[..., 0]
        if depth_values.ndim != 4:
            raise ValueError(f"Expected depth_values shape (B,V,H,W), got {tuple(depth_values.shape)}")
        bsz, num_views, height, width = depth_values.shape
        if num_views != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {num_views}")

        depth = depth_values.to(dtype=torch.float32)
        intrinsics = depth_intrinsics.to(device=depth.device, dtype=torch.float32)
        extrinsics = depth_extrinsics.to(device=depth.device, dtype=torch.float32)
        valid = torch.isfinite(depth) & (depth >= self.depth_min_m) & (depth <= self.depth_max_m)
        if depth_valid_mask is not None:
            if depth_valid_mask.ndim == 5 and depth_valid_mask.shape[-1] == 1:
                depth_valid_mask = depth_valid_mask[..., 0]
            valid = valid & depth_valid_mask.to(device=depth.device).bool()
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        depth, valid = self._apply_depth_ablation(depth, valid)

        flat_indices = self._regular_sample_indices(height, width, depth.device)
        y_idx = torch.div(flat_indices, width, rounding_mode="floor")
        x_idx = flat_indices % width
        sampled_depth = depth.reshape(bsz, num_views, height * width).index_select(-1, flat_indices)
        sampled_valid = valid.reshape(bsz, num_views, height * width).index_select(-1, flat_indices)
        x = x_idx.to(torch.float32).view(1, 1, -1).expand(bsz, num_views, -1)
        y = y_idx.to(torch.float32).view(1, 1, -1).expand(bsz, num_views, -1)

        fx = self._safe_signed_focal(intrinsics[:, :, 0, 0].unsqueeze(-1))
        fy = self._safe_signed_focal(intrinsics[:, :, 1, 1].unsqueeze(-1))
        cx = intrinsics[:, :, 0, 2].unsqueeze(-1)
        cy = intrinsics[:, :, 1, 2].unsqueeze(-1)

        z_cam = sampled_depth
        x_cam = (x - cx) * z_cam / fx
        y_cam = (y - cy) * z_cam / fy
        ones = torch.ones_like(z_cam)
        xyz1_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)
        xyz1_base = torch.einsum("bvij,bvnj->bvni", extrinsics, xyz1_cam)
        xyz_base = xyz1_base[..., :3]

        if ee_pos is None:
            ee_xyz = torch.zeros(bsz, 1, 1, 3, device=depth.device, dtype=torch.float32)
        else:
            ee_xyz = ee_pos[..., :3].to(device=depth.device, dtype=torch.float32).view(bsz, 1, 1, 3)
        rel_xyz = xyz_base - ee_xyz
        radius = torch.linalg.norm(rel_xyz, dim=-1, keepdim=True)

        u_norm = (x / max(width - 1, 1)).unsqueeze(-1)
        v_norm = (y / max(height - 1, 1)).unsqueeze(-1)
        view_ids = torch.linspace(0, 1, steps=num_views, device=depth.device, dtype=torch.float32)
        view_ids = view_ids.view(1, num_views, 1, 1).expand(bsz, num_views, self.num_points_per_view, 1)
        valid_f = sampled_valid.to(torch.float32).unsqueeze(-1)

        features = torch.cat(
            [
                xyz_base,
                rel_xyz,
                radius,
                z_cam.unsqueeze(-1),
                valid_f,
                u_norm,
                v_norm,
                view_ids,
            ],
            dim=-1,
        )
        return features.reshape(bsz, self.depth_num_tokens, 12)

    def _regular_sample_indices(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        total = height * width
        if self.num_points_per_view == total:
            return torch.arange(total, device=device)
        if self.num_points_per_view > total:
            base = torch.arange(total, device=device)
            repeats = (self.num_points_per_view + total - 1) // total
            return base.repeat(repeats)[: self.num_points_per_view]
        return torch.linspace(0, total - 1, steps=self.num_points_per_view, device=device).round().long()

    @staticmethod
    def _safe_signed_focal(focal: torch.Tensor) -> torch.Tensor:
        eps = torch.full_like(focal, 1e-6)
        sign = torch.where(focal < 0, -torch.ones_like(focal), torch.ones_like(focal))
        return torch.where(focal.abs() < eps, sign * eps, focal)

    def _apply_depth_ablation(self, depth: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mode = str(getattr(self, "ablation_mode", "none") or "none").lower()
        if mode not in ("shuffle_depth", "shuffle_pixels", "shuffle_depth_pixels"):
            return depth, valid

        bsz, num_views, height, width = depth.shape
        total = height * width
        depth_flat = depth.reshape(bsz, num_views, total).clone()
        valid_flat = valid.reshape(bsz, num_views, total).clone()
        generator = torch.Generator(device=depth.device)
        generator.manual_seed(int(getattr(self, "shuffle_seed", 0)))
        for bidx in range(bsz):
            for vidx in range(num_views):
                perm = torch.randperm(total, generator=generator, device=depth.device)
                depth_flat[bidx, vidx] = depth_flat[bidx, vidx, perm]
                valid_flat[bidx, vidx] = valid_flat[bidx, vidx, perm]
        return depth_flat.reshape_as(depth), valid_flat.reshape_as(valid)

    def _apply_ablation(self, features: torch.Tensor) -> torch.Tensor:
        mode = str(getattr(self, "ablation_mode", "none") or "none").lower()
        if mode in ("", "none", "shuffle_depth", "shuffle_pixels", "shuffle_depth_pixels"):
            return features
        if mode in ("null", "zero"):
            return torch.zeros_like(features)
        generator = torch.Generator(device=features.device)
        generator.manual_seed(int(getattr(self, "shuffle_seed", 0)))
        perm = torch.randperm(features.shape[1], generator=generator, device=features.device)
        if mode in ("shuffle_tokens", "shuffled"):
            return features[:, perm, :]
        if mode in ("shuffle_geometry", "shuffle_xyz", "shuffle_depth_geometry"):
            out = features.clone()
            # Shuffle base XYZ, EE-relative XYZ, radius, and camera depth while
            # keeping UV/view identifiers fixed. This corrupts geometry without
            # destroying token position metadata.
            out[..., :8] = features[:, perm, :8]
            return out
        raise ValueError(f"Unknown depth ablation mode: {mode}")


if __name__ == "__main__":
    model = DensePointDepthTokenEncoder(llm_dim=64, hidden_dim=32, num_points_per_view=128)
    depth = torch.rand(2, 2, 64, 64) + 0.2
    k = torch.eye(3).view(1, 1, 3, 3).expand(2, 2, -1, -1).clone()
    k[:, :, 0, 0] = 60
    k[:, :, 1, 1] = 60
    k[:, :, 0, 2] = 32
    k[:, :, 1, 2] = 32
    t = torch.eye(4).view(1, 1, 4, 4).expand(2, 2, -1, -1).clone()
    out = model(depth, k, t, ee_pos=torch.zeros(2, 3))
    assert out.shape == (2, 256, 64), out.shape
    print("DensePointDepthTokenEncoder smoke passed", tuple(out.shape))
