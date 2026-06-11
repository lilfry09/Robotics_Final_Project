"""Lightweight depth-to-token encoder for DepthVLA-OFT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


GEOMETRY_CONTINUOUS_FEATURE_NAMES = ("X_base", "Y_base", "Z_base", "z_camera")


class LightweightDepthTokenEncoder(nn.Module):
    """Encodes metric RGB-D geometry into LLM-space prefix tokens.

    Inputs are raw, unrotated depth maps and camera matrices. The encoder computes
    base/world-frame XYZ from camera-local metric depth, pools the resulting
    geometry into a coarse grid, flips the grid by 180 degrees to match LIBERO's
    rotated RGB convention, and embeds each grid cell into the LLM hidden size.
    """

    def __init__(
        self,
        llm_dim: int,
        hidden_dim: int = 256,
        grid_size: int = 4,
        depth_min_m: float = 0.01,
        depth_max_m: float = 5.0,
        num_views: int = 2,
        geometry_norm: str = "none",
        geometry_clip: float | None = 5.0,
        enable_summary: bool = False,
        summary_repr: str = "base_xyz",
        summary_pool: str = "meanmax",
    ) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self.num_views = num_views
        self.depth_num_tokens = num_views * grid_size * grid_size
        self.geometry_norm = geometry_norm
        self.geometry_clip = geometry_clip
        self.enable_summary = enable_summary
        self.summary_repr = summary_repr
        self.summary_pool = summary_pool

        self.encoder = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )
        self.summary_mlp = None
        self.set_encoder = None
        self.set_summary_mlp = None
        self.summary_feature_dim = 8
        if enable_summary:
            if summary_repr == "base_xyz":
                # Action-side summary conditioning: compact depth representation for
                # the action head. This is not appended to the VLM/LLM prefix.
                self.summary_mlp = nn.Sequential(
                    nn.LayerNorm(llm_dim * num_views),
                    nn.Linear(llm_dim * num_views, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, llm_dim),
                    nn.LayerNorm(llm_dim),
                )
            elif summary_repr == "ee_relative_set_v2":
                if summary_pool != "meanmax":
                    raise ValueError(f"Unknown summary_pool for ee_relative_set_v2: {summary_pool}")
                # Robot-centric Deep Sets / PointNet-style summary. Each coarse
                # geometry cell is encoded independently, then pooled without
                # depending on cell order.
                self.summary_feature_dim = 9
                self.set_encoder = nn.Sequential(
                    nn.Linear(self.summary_feature_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, llm_dim),
                    nn.LayerNorm(llm_dim),
                )
                self.set_summary_mlp = nn.Sequential(
                    nn.LayerNorm(llm_dim * 2),
                    nn.Linear(llm_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, llm_dim),
                    nn.LayerNorm(llm_dim),
                )
            else:
                raise ValueError(f"Unknown summary_repr: {summary_repr}")
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.ablation_mode = "none"
        self.shuffle_seed = 0
        self.register_buffer("geometry_norm_mean", torch.zeros(4, dtype=torch.float32), persistent=False)
        self.register_buffer("geometry_norm_std", torch.ones(4, dtype=torch.float32), persistent=False)

    def set_geometry_normalization(
        self,
        stats: dict | None,
        geometry_norm: str = "none",
        geometry_clip: float | None = 5.0,
        enable_summary: bool = False,
        summary_repr: str = "base_xyz",
        summary_pool: str = "meanmax",
    ) -> None:
        self.geometry_norm = geometry_norm
        self.geometry_clip = geometry_clip
        if geometry_norm == "none":
            return
        if geometry_norm != "dataset_std":
            raise ValueError(f"Unknown geometry_norm mode: {geometry_norm}")
        if stats is None:
            raise ValueError("geometry_norm='dataset_std' requires dataset geometry statistics")
        mean = torch.tensor(stats["mean"], dtype=torch.float32, device=self.geometry_norm_mean.device)
        std = torch.tensor(stats["std"], dtype=torch.float32, device=self.geometry_norm_std.device)
        if mean.shape != (4,) or std.shape != (4,):
            raise ValueError(f"Expected 4D geometry stats, got mean={tuple(mean.shape)}, std={tuple(std.shape)}")
        self.geometry_norm_mean.copy_(mean)
        self.geometry_norm_std.copy_(std)

    def forward(
        self,
        depth_values: torch.Tensor,
        depth_intrinsics: torch.Tensor,
        depth_extrinsics: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return geometry tokens with shape ``(B, V * grid_size^2, llm_dim)``.

        Args:
            depth_values: Metric depth maps, shape ``(B, V, H, W)``.
            depth_intrinsics: Camera intrinsics, shape ``(B, V, 3, 3)``.
            depth_extrinsics: Camera-to-base/world transforms, shape ``(B, V, 4, 4)``.
            depth_valid_mask: Optional validity mask, shape ``(B, V, H, W)``.
        """
        pooled = self.compute_geometry_features(depth_values, depth_intrinsics, depth_extrinsics, depth_valid_mask)
        pooled = self._normalize_geometry_features(pooled)
        pooled = self._apply_ablation(pooled)
        tokens = self.encoder(pooled.to(dtype=next(self.encoder.parameters()).dtype))
        return self.alpha.to(tokens.dtype) * tokens


    def forward_summary(
        self,
        depth_values: torch.Tensor,
        depth_intrinsics: torch.Tensor,
        depth_extrinsics: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
        ee_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return compact depth summary with shape (B, llm_dim).

        The same metric geometry features used by the token path are encoded per
        grid cell, then mean-pooled per view. The two view summaries are
        concatenated and projected to a single action-side summary embedding.
        """
        if self.summary_repr == "ee_relative_set_v2":
            return self._forward_ee_relative_set_summary(
                depth_values=depth_values,
                depth_intrinsics=depth_intrinsics,
                depth_extrinsics=depth_extrinsics,
                depth_valid_mask=depth_valid_mask,
                ee_pos=ee_pos,
            )
        if self.summary_mlp is None:
            raise ValueError("forward_summary requires LightweightDepthTokenEncoder(enable_summary=True)")
        pooled = self.compute_geometry_features(depth_values, depth_intrinsics, depth_extrinsics, depth_valid_mask)
        pooled = self._normalize_geometry_features(pooled)
        pooled = self._apply_ablation(pooled)
        token_features = self.encoder(pooled.to(dtype=next(self.encoder.parameters()).dtype))
        bsz = token_features.shape[0]
        token_features = token_features.reshape(bsz, self.num_views, self.grid_size * self.grid_size, self.llm_dim)
        view_summary = token_features.mean(dim=2)
        summary_input = view_summary.reshape(bsz, self.num_views * self.llm_dim)
        summary = self.summary_mlp(summary_input)
        return self.alpha.to(summary.dtype) * summary


    def _forward_ee_relative_set_summary(
        self,
        depth_values: torch.Tensor,
        depth_intrinsics: torch.Tensor,
        depth_extrinsics: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
        ee_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.set_encoder is None or self.set_summary_mlp is None:
            raise ValueError("ee_relative_set_v2 requires LightweightDepthTokenEncoder(enable_summary=True)")
        pooled = self.compute_geometry_features(depth_values, depth_intrinsics, depth_extrinsics, depth_valid_mask)
        set_features = self._make_ee_relative_features(pooled, ee_pos)
        set_features = self._apply_ablation(set_features)
        point_embeddings = self.set_encoder(set_features.to(dtype=next(self.set_encoder.parameters()).dtype))
        mean_pool = point_embeddings.mean(dim=1)
        max_pool = point_embeddings.max(dim=1).values
        summary = self.set_summary_mlp(torch.cat([mean_pool, max_pool], dim=-1))
        return self.alpha.to(summary.dtype) * summary

    def compute_geometry_features(
        self,
        depth_values: torch.Tensor,
        depth_intrinsics: torch.Tensor,
        depth_extrinsics: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return pooled pre-MLP features with shape ``(B, V * grid_size^2, 8)``."""
        if depth_values.ndim == 5 and depth_values.shape[-1] == 1:
            depth_values = depth_values[..., 0]
        if depth_values.ndim != 4:
            raise ValueError(f"Expected depth_values with shape (B,V,H,W), got {tuple(depth_values.shape)}")

        bsz, num_views, height, width = depth_values.shape
        if num_views != self.num_views:
            raise ValueError(f"Expected {self.num_views} depth views, got {num_views}")

        depth = depth_values.to(dtype=torch.float32)
        intrinsics = depth_intrinsics.to(device=depth.device, dtype=torch.float32)
        extrinsics = depth_extrinsics.to(device=depth.device, dtype=torch.float32)

        valid = torch.isfinite(depth) & (depth >= self.depth_min_m) & (depth <= self.depth_max_m)
        if depth_valid_mask is not None:
            if depth_valid_mask.ndim == 5 and depth_valid_mask.shape[-1] == 1:
                depth_valid_mask = depth_valid_mask[..., 0]
            valid = valid & depth_valid_mask.to(device=depth.device).bool()
        depth = torch.where(valid, depth, torch.zeros_like(depth))

        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=depth.device, dtype=torch.float32),
            torch.arange(width, device=depth.device, dtype=torch.float32),
            indexing="ij",
        )
        x_coords = x_coords.view(1, 1, height, width)
        y_coords = y_coords.view(1, 1, height, width)

        fx = intrinsics[:, :, 0, 0].view(bsz, num_views, 1, 1).clamp_min(1e-6)
        fy = intrinsics[:, :, 1, 1].view(bsz, num_views, 1, 1).clamp_min(1e-6)
        cx = intrinsics[:, :, 0, 2].view(bsz, num_views, 1, 1)
        cy = intrinsics[:, :, 1, 2].view(bsz, num_views, 1, 1)

        z_cam = depth
        x_cam = (x_coords - cx) * z_cam / fx
        y_cam = (y_coords - cy) * z_cam / fy
        ones = torch.ones_like(z_cam)
        xyz1_cam = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)
        xyz1_base = torch.einsum("bvij,bvhwj->bvhwi", extrinsics, xyz1_cam)
        xyz_base = xyz1_base[..., :3]

        u_norm = (x_coords / max(width - 1, 1)).expand(bsz, num_views, height, width)
        v_norm = (y_coords / max(height - 1, 1)).expand(bsz, num_views, height, width)
        view_ids = torch.linspace(0, 1, steps=num_views, device=depth.device, dtype=torch.float32)
        view_ids = view_ids.view(1, num_views, 1, 1).expand(bsz, num_views, height, width)

        features = torch.cat(
            [
                xyz_base,
                z_cam.unsqueeze(-1),
                valid.to(torch.float32).unsqueeze(-1),
                u_norm.unsqueeze(-1),
                v_norm.unsqueeze(-1),
                view_ids.unsqueeze(-1),
            ],
            dim=-1,
        )
        pooled = self._valid_average_pool(features, valid)

        # LIBERO RGB is rotated by 180 degrees at policy input time. We compute
        # geometry from raw camera pixels, then flip token grid to preserve
        # coarse spatial correspondence with the rotated RGB patch order.
        pooled = torch.flip(pooled, dims=[2, 3])
        pooled = pooled.reshape(bsz, num_views * self.grid_size * self.grid_size, 8)
        return pooled


    def _make_ee_relative_features(self, features: torch.Tensor, ee_pos: torch.Tensor | None) -> torch.Tensor:
        """Convert base-frame XYZ cells into robot-centric set features.

        Output feature order: dx, dy, dz, radius, z_camera, valid, u_norm, v_norm, view_id.
        """
        if ee_pos is None:
            raise ValueError("summary_repr='ee_relative_set_v2' requires raw end-effector position ee_pos with shape (B,3)")
        if ee_pos.ndim == 1:
            ee_pos = ee_pos.unsqueeze(0)
        if ee_pos.shape[-1] < 3:
            raise ValueError(f"Expected ee_pos last dimension >= 3, got {tuple(ee_pos.shape)}")
        ee_xyz = ee_pos[..., :3].to(device=features.device, dtype=torch.float32).view(features.shape[0], 1, 3)
        xyz = features[..., :3].to(torch.float32)
        dxyz = xyz - ee_xyz
        radius = torch.linalg.norm(dxyz, dim=-1, keepdim=True)
        z_camera = features[..., 3:4].to(torch.float32)
        rest = features[..., 4:8].to(torch.float32)
        return torch.cat([dxyz, radius, z_camera, rest], dim=-1)

    def _normalize_geometry_features(self, features: torch.Tensor) -> torch.Tensor:
        mode = getattr(self, "geometry_norm", "none")
        if mode in (None, "", "none"):
            return features
        if mode != "dataset_std":
            raise ValueError(f"Unknown geometry_norm mode: {mode}")

        out = features.clone()
        mean = self.geometry_norm_mean.to(device=out.device, dtype=torch.float32).view(1, 1, 4)
        std = self.geometry_norm_std.to(device=out.device, dtype=torch.float32).view(1, 1, 4)
        normalized = (out[..., :4].to(torch.float32) - mean) / (std + 1e-6)
        if self.geometry_clip is not None and self.geometry_clip > 0:
            normalized = normalized.clamp(-float(self.geometry_clip), float(self.geometry_clip))
        out[..., :4] = normalized.to(out.dtype)
        return out

    def _apply_ablation(self, features: torch.Tensor) -> torch.Tensor:
        mode = getattr(self, "ablation_mode", "none")
        if mode is None:
            mode = "null"
        mode = str(mode).lower()
        if mode in ("", "none"):
            return features
        if mode in ("null", "zero"):
            return torch.zeros_like(features)
        if mode in ("shuffle_tokens", "shuffled"):
            generator = torch.Generator(device=features.device)
            generator.manual_seed(int(getattr(self, "shuffle_seed", 0)))
            perm = torch.randperm(features.shape[1], generator=generator, device=features.device)
            return features[:, perm, :]
        raise ValueError(f"Unknown depth ablation mode: {mode}")

    def _valid_average_pool(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        bsz, num_views, height, width, feat_dim = features.shape
        flat_features = features.permute(0, 1, 4, 2, 3).reshape(bsz * num_views, feat_dim, height, width)
        flat_valid = valid.to(torch.float32).reshape(bsz * num_views, 1, height, width)

        weighted = F.adaptive_avg_pool2d(flat_features * flat_valid, (self.grid_size, self.grid_size))
        counts = F.adaptive_avg_pool2d(flat_valid, (self.grid_size, self.grid_size)).clamp_min(1e-6)
        pooled = weighted / counts
        pooled = pooled.reshape(bsz, num_views, feat_dim, self.grid_size, self.grid_size)
        return pooled.permute(0, 1, 3, 4, 2)
