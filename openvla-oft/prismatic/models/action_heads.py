"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        use_depth_conditioning=False,
        depth_fusion_type="hidden_film",
        depth_fusion_gate_init=0.001,
        depth_adapter_hidden_dim=256,
        spatial_aux_output_dim=3,
        depth_hidden_delta_clip=0.0,
        depth_action_residual_clip=0.0,
        depth_keypose_residual_weight=0.0,
        depth_keypose_residual_clip=0.0,
        depth_point_action_weight=0.0,
        depth_point_action_clip=0.0,
        depth_waypoint_action_weight=0.0,
        depth_waypoint_action_clip=0.0,
        depth_waypoint_action_scale=1.0,
        depth_waypoint_action_chunk_len=1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.use_depth_conditioning = use_depth_conditioning
        self.depth_fusion_type = depth_fusion_type
        self.spatial_aux_output_dim = int(spatial_aux_output_dim)
        self.depth_hidden_delta_clip = float(depth_hidden_delta_clip or 0.0)
        self.depth_action_residual_clip = float(depth_action_residual_clip or 0.0)
        self.depth_keypose_residual_weight = float(depth_keypose_residual_weight or 0.0)
        self.depth_keypose_residual_clip = float(depth_keypose_residual_clip or 0.0)
        self.depth_point_action_weight = float(depth_point_action_weight or 0.0)
        self.depth_point_action_clip = float(depth_point_action_clip or 0.0)
        self.depth_waypoint_action_weight = float(depth_waypoint_action_weight or 0.0)
        self.depth_waypoint_action_clip = float(depth_waypoint_action_clip or 0.0)
        self.depth_waypoint_action_scale = float(depth_waypoint_action_scale or 1.0)
        self.depth_waypoint_action_chunk_len = int(depth_waypoint_action_chunk_len or 1)
        self.projected_heatmap_size = self._infer_projected_heatmap_size(self.spatial_aux_output_dim)
        self.projected_heatmap_num_views = 2
        self.depth_keypose_residual_feature_dim = self.spatial_aux_output_dim
        if self.projected_heatmap_size is not None:
            self.depth_keypose_residual_feature_dim += self.projected_heatmap_num_views * 2
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )
        if use_depth_conditioning:
            self.depth_fusion_gate = nn.Parameter(torch.tensor(float(depth_fusion_gate_init)))
            self.depth_keypose_action_residual = None
            self.depth_point_score_head = None
            self.depth_point_action_residual = None
            if depth_fusion_type == "hidden_film":
                # Action-side depth fusion: depth never changes the VLM prefix; it only modulates action hidden states.
                self.depth_context = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, input_dim * 2),
                )
                nn.init.zeros_(self.depth_context[-1].weight)
                nn.init.zeros_(self.depth_context[-1].bias)
                self.depth_action_residual = None
            elif depth_fusion_type == "action_summary_aux":
                # Action-side summary conditioning: depth summary shifts the
                # action-token hidden states, without touching the RGB prefix.
                self.depth_context = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, depth_adapter_hidden_dim),
                    nn.GELU(),
                    nn.Linear(depth_adapter_hidden_dim, input_dim),
                )
                nn.init.zeros_(self.depth_context[-1].weight)
                nn.init.zeros_(self.depth_context[-1].bias)
                self.depth_action_residual = None
            elif depth_fusion_type == "object_query":
                # Language/action-conditioned depth fusion. The action token
                # hidden state already carries instruction context, so use it
                # to form object/target/contact queries over metric depth tokens.
                num_heads = 8 if input_dim % 8 == 0 else 1
                self.depth_query_embeddings = nn.Parameter(torch.zeros(3, input_dim))
                self.depth_query_norm = nn.LayerNorm(input_dim)
                self.depth_query_proj = nn.Linear(input_dim, input_dim)
                self.depth_language_query_norm = nn.LayerNorm(input_dim)
                self.depth_language_query_proj = nn.Linear(input_dim, input_dim)
                self.depth_token_norm = nn.LayerNorm(input_dim)
                self.depth_cross_attn = nn.MultiheadAttention(input_dim, num_heads=num_heads, batch_first=True)
                self.depth_context = nn.Sequential(
                    nn.LayerNorm(input_dim * 3),
                    nn.Linear(input_dim * 3, depth_adapter_hidden_dim),
                    nn.GELU(),
                    nn.Linear(depth_adapter_hidden_dim, input_dim),
                )
                self.object_query_spatial_heads = nn.ModuleDict(
                    {
                        "ee_to_object_xyz": nn.Sequential(
                            nn.LayerNorm(input_dim),
                            nn.Linear(input_dim, depth_adapter_hidden_dim),
                            nn.GELU(),
                            nn.Linear(depth_adapter_hidden_dim, 3),
                        ),
                        "object_to_target_xyz": nn.Sequential(
                            nn.LayerNorm(input_dim),
                            nn.Linear(input_dim, depth_adapter_hidden_dim),
                            nn.GELU(),
                            nn.Linear(depth_adapter_hidden_dim, 3),
                        ),
                        "gripper_to_contact_distance": nn.Sequential(
                            nn.LayerNorm(input_dim),
                            nn.Linear(input_dim, depth_adapter_hidden_dim),
                            nn.GELU(),
                            nn.Linear(depth_adapter_hidden_dim, 1),
                        ),
                        "distance_bin": nn.Sequential(
                            nn.LayerNorm(input_dim),
                            nn.Linear(input_dim, depth_adapter_hidden_dim),
                            nn.GELU(),
                            nn.Linear(depth_adapter_hidden_dim, self.spatial_aux_output_dim),
                        ),
                        "relative_z_bin": nn.Sequential(
                            nn.LayerNorm(input_dim),
                            nn.Linear(input_dim, depth_adapter_hidden_dim),
                            nn.GELU(),
                            nn.Linear(depth_adapter_hidden_dim, self.spatial_aux_output_dim),
                        ),
                        "absolute_keypose": nn.Sequential(
                            nn.LayerNorm(input_dim * 3),
                            nn.Linear(input_dim * 3, depth_adapter_hidden_dim),
                            nn.GELU(),
                            nn.Linear(depth_adapter_hidden_dim, self.spatial_aux_output_dim),
                        ),
                    }
                )
                self.depth_point_score_head = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, depth_adapter_hidden_dim),
                    nn.GELU(),
                    nn.Linear(depth_adapter_hidden_dim, 1),
                )
                nn.init.normal_(self.depth_query_embeddings, mean=0.0, std=0.02)
                nn.init.zeros_(self.depth_query_proj.weight)
                nn.init.zeros_(self.depth_query_proj.bias)
                nn.init.zeros_(self.depth_language_query_proj.weight)
                nn.init.zeros_(self.depth_language_query_proj.bias)
                nn.init.zeros_(self.depth_context[-1].weight)
                nn.init.zeros_(self.depth_context[-1].bias)
                self.depth_action_residual = None
                if self.depth_keypose_residual_weight > 0:
                    keypose_input_dim = input_dim + input_dim * 3 + self.depth_keypose_residual_feature_dim
                    self.depth_keypose_action_residual = nn.Sequential(
                        nn.LayerNorm(keypose_input_dim),
                        nn.Linear(keypose_input_dim, depth_adapter_hidden_dim),
                        nn.GELU(),
                        nn.Linear(depth_adapter_hidden_dim, action_dim),
                    )
                    nn.init.zeros_(self.depth_keypose_action_residual[-1].weight)
                    nn.init.zeros_(self.depth_keypose_action_residual[-1].bias)
                if self.depth_point_action_weight > 0:
                    point_action_input_dim = input_dim + 3
                    self.depth_point_action_residual = nn.Sequential(
                        nn.LayerNorm(point_action_input_dim),
                        nn.Linear(point_action_input_dim, depth_adapter_hidden_dim),
                        nn.GELU(),
                        nn.Linear(depth_adapter_hidden_dim, 3),
                    )
                self.last_depth_query_attention = None
                self.last_depth_point_logits = None
                self.last_depth_point_xyz = None
                self.last_depth_waypoint_xyz_action = None
                self.last_depth_waypoint_chunk_xyz_action = None
            elif depth_fusion_type == "action_residual":
                self.depth_context = None
                adapter_input_dim = input_dim * 2
                self.depth_action_residual = nn.Sequential(
                    nn.LayerNorm(adapter_input_dim),
                    nn.Linear(adapter_input_dim, depth_adapter_hidden_dim),
                    nn.GELU(),
                    nn.Linear(depth_adapter_hidden_dim, action_dim),
                )
                nn.init.zeros_(self.depth_action_residual[-1].weight)
                nn.init.zeros_(self.depth_action_residual[-1].bias)
            else:
                raise ValueError(f"Unknown depth_fusion_type: {depth_fusion_type}")
            self.spatial_head = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, depth_adapter_hidden_dim),
                nn.GELU(),
                nn.Linear(depth_adapter_hidden_dim, self.spatial_aux_output_dim),
            )
        else:
            self.depth_context = None
            self.depth_action_residual = None
            self.depth_keypose_action_residual = None
            self.depth_point_score_head = None
            self.depth_point_action_residual = None
            self.last_depth_waypoint_xyz_action = None
            self.last_depth_waypoint_chunk_xyz_action = None
            self.spatial_head = None
            self.object_query_spatial_heads = None

    @staticmethod
    def _infer_projected_heatmap_size(output_dim: int) -> int | None:
        output_dim = int(output_dim)
        if output_dim < 32 or output_dim % 2 != 0:
            return None
        per_view = output_dim // 2
        size = int(math.sqrt(per_view))
        if size * size == per_view:
            return size
        return None

    def _projected_heatmap_logits(self, spatial_flat: torch.Tensor) -> torch.Tensor:
        if self.projected_heatmap_size is None:
            raise ValueError(
                f"spatial_aux_output_dim={self.spatial_aux_output_dim} is not a two-view square heatmap"
            )
        return spatial_flat.reshape(
            spatial_flat.shape[0],
            self.projected_heatmap_num_views,
            self.projected_heatmap_size,
            self.projected_heatmap_size,
        )

    def _projected_heatmap_softargmax(self, spatial_flat: torch.Tensor) -> torch.Tensor:
        logits = self._projected_heatmap_logits(spatial_flat).float()
        batch_size, num_views, height, width = logits.shape
        probs = torch.softmax(logits.reshape(batch_size, num_views, -1), dim=-1)
        xs = torch.linspace(-1.0, 1.0, width, device=logits.device, dtype=probs.dtype)
        ys = torch.linspace(-1.0, 1.0, height, device=logits.device, dtype=probs.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        x = (probs * xx.reshape(1, 1, -1)).sum(dim=-1)
        y = (probs * yy.reshape(1, 1, -1)).sum(dim=-1)
        coords = torch.stack([x, y], dim=-1).reshape(batch_size, num_views * 2)
        return coords.to(dtype=spatial_flat.dtype)

    def _spatial_residual_features(self, spatial_flat: torch.Tensor) -> torch.Tensor:
        if self.projected_heatmap_size is None:
            return spatial_flat
        coords = self._projected_heatmap_softargmax(spatial_flat)
        return torch.cat([spatial_flat, coords], dim=-1)

    def pool_depth_context(self, depth_context):
        if depth_context is None:
            return None
        if depth_context.ndim == 3:
            depth_context = depth_context.mean(dim=1)
        if depth_context.ndim != 2:
            raise ValueError(f"Expected depth_context with shape (B,D) or (B,T,D), got {tuple(depth_context.shape)}")
        return depth_context

    def _pool_depth_context(self, depth_context):
        return self.pool_depth_context(depth_context)

    def object_query_attended_tokens(self, actions_hidden_states, depth_context, query_context=None):
        if depth_context.ndim != 3:
            raise ValueError(
                f"object_query depth fusion expects per-cell depth tokens with shape (B,T,D), got {tuple(depth_context.shape)}"
            )
        batch_size = actions_hidden_states.shape[0]
        action_context = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM, -1).mean(dim=(1, 2))
        query_seed = self.depth_query_proj(self.depth_query_norm(action_context))
        if query_context is not None:
            if query_context.ndim == 3:
                query_context = query_context.mean(dim=1)
            if query_context.ndim != 2:
                raise ValueError(
                    f"object_query query_context expects shape (B,D) or (B,T,D), got {tuple(query_context.shape)}"
                )
            query_context = query_context.to(device=actions_hidden_states.device, dtype=actions_hidden_states.dtype)
            query_seed = query_seed + self.depth_language_query_proj(self.depth_language_query_norm(query_context))
        queries = query_seed.unsqueeze(1) + self.depth_query_embeddings.to(
            device=actions_hidden_states.device, dtype=actions_hidden_states.dtype
        ).unsqueeze(0)
        depth_tokens = self.depth_token_norm(depth_context.to(device=actions_hidden_states.device, dtype=actions_hidden_states.dtype))
        attended, attn_weights = self.depth_cross_attn(queries, depth_tokens, depth_tokens, need_weights=True)
        self.last_depth_query_attention = attn_weights.detach()
        return attended

    def object_query_context(self, actions_hidden_states, depth_context, query_context=None):
        attended = self.object_query_attended_tokens(actions_hidden_states, depth_context, query_context=query_context)
        batch_size = actions_hidden_states.shape[0]
        return attended.reshape(batch_size, -1)

    def condition_action_hidden_states(self, actions_hidden_states, depth_context=None, query_context=None):
        if not self.use_depth_conditioning or depth_context is None:
            return actions_hidden_states
        if self.depth_fusion_type not in ("hidden_film", "action_summary_aux", "object_query"):
            return actions_hidden_states
        gate = self.depth_fusion_gate.to(dtype=actions_hidden_states.dtype)
        if self.depth_fusion_type == "object_query":
            depth_query_context = self.object_query_context(actions_hidden_states, depth_context, query_context=query_context)
            delta_h = self.depth_context(depth_query_context).unsqueeze(1)
            if self.depth_hidden_delta_clip > 0:
                delta_h = delta_h.clamp(-self.depth_hidden_delta_clip, self.depth_hidden_delta_clip)
            return actions_hidden_states + gate * delta_h

        depth_context = self._pool_depth_context(depth_context).to(
            device=actions_hidden_states.device, dtype=actions_hidden_states.dtype
        )
        if self.depth_fusion_type == "action_summary_aux":
            delta_h = self.depth_context(depth_context).unsqueeze(1)
            if self.depth_hidden_delta_clip > 0:
                delta_h = delta_h.clamp(-self.depth_hidden_delta_clip, self.depth_hidden_delta_clip)
            return actions_hidden_states + gate * delta_h
        scale_shift = self.depth_context(depth_context).unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        if self.depth_hidden_delta_clip > 0:
            scale = scale.clamp(-self.depth_hidden_delta_clip, self.depth_hidden_delta_clip)
            shift = shift.clamp(-self.depth_hidden_delta_clip, self.depth_hidden_delta_clip)
        return actions_hidden_states + gate * (actions_hidden_states * scale + shift)

    def predict_action_residual(self, actions_hidden_states, depth_context=None):
        if not self.use_depth_conditioning or self.depth_fusion_type != "action_residual" or depth_context is None:
            return None
        batch_size = actions_hidden_states.shape[0]
        depth_context = self._pool_depth_context(depth_context).to(
            device=actions_hidden_states.device, dtype=actions_hidden_states.dtype
        )
        action_context = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM, -1).mean(dim=2)
        depth_context = depth_context.unsqueeze(1).expand(-1, NUM_ACTIONS_CHUNK, -1)
        adapter_input = torch.cat([action_context, depth_context], dim=-1)
        gate = self.depth_fusion_gate.to(dtype=actions_hidden_states.dtype)
        residual = gate * self.depth_action_residual(adapter_input)
        if self.depth_action_residual_clip > 0:
            residual = residual.clamp(-self.depth_action_residual_clip, self.depth_action_residual_clip)
        return residual

    def predict_keypose_action_residual(self, actions_hidden_states, depth_context=None, query_context=None):
        if (
            not self.use_depth_conditioning
            or self.depth_fusion_type != "object_query"
            or self.depth_keypose_action_residual is None
            or depth_context is None
        ):
            return None
        batch_size = actions_hidden_states.shape[0]
        attended = self.object_query_attended_tokens(
            actions_hidden_states, depth_context, query_context=query_context
        )
        attended_context = attended.reshape(batch_size, -1)
        keypose = self.object_query_spatial_heads["absolute_keypose"](attended_context)
        keypose_features = self._spatial_residual_features(keypose)
        action_context = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM, -1).mean(dim=2)
        keypose_context = torch.cat([attended_context, keypose_features], dim=-1).unsqueeze(1)
        keypose_context = keypose_context.expand(-1, NUM_ACTIONS_CHUNK, -1)
        adapter_input = torch.cat([action_context, keypose_context], dim=-1)
        gate = self.depth_fusion_gate.to(dtype=actions_hidden_states.dtype)
        residual = gate * self.depth_keypose_residual_weight * self.depth_keypose_action_residual(adapter_input)
        residual_clip = self.depth_keypose_residual_clip or self.depth_action_residual_clip
        if residual_clip > 0:
            residual = residual.clamp(-residual_clip, residual_clip)
        return residual

    def predict_point_keypose_xyz(self, actions_hidden_states, depth_context=None, depth_point_features=None, query_context=None):
        if (
            not self.use_depth_conditioning
            or self.depth_fusion_type != "object_query"
            or self.depth_point_score_head is None
            or depth_context is None
            or depth_point_features is None
        ):
            raise ValueError("Point keypose prediction requires object_query depth_context and depth_point_features")
        if depth_context.ndim != 3:
            raise ValueError(f"Expected depth_context shape (B,T,D), got {tuple(depth_context.shape)}")
        if depth_point_features.ndim != 3 or depth_point_features.shape[:2] != depth_context.shape[:2]:
            raise ValueError(
                "Expected depth_point_features shape (B,T,F) aligned with depth_context; "
                f"got {tuple(depth_point_features.shape)} vs {tuple(depth_context.shape)}"
            )
        batch_size = actions_hidden_states.shape[0]
        action_context = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM, -1).mean(dim=(1, 2))
        query_seed = self.depth_query_proj(self.depth_query_norm(action_context))
        if query_context is not None:
            if query_context.ndim == 3:
                query_context = query_context.mean(dim=1)
            if query_context.ndim != 2:
                raise ValueError(
                    f"point keypose query_context expects shape (B,D) or (B,T,D), got {tuple(query_context.shape)}"
                )
            query_context = query_context.to(device=actions_hidden_states.device, dtype=actions_hidden_states.dtype)
            query_seed = query_seed + self.depth_language_query_proj(self.depth_language_query_norm(query_context))

        depth_tokens = self.depth_token_norm(depth_context.to(device=actions_hidden_states.device, dtype=actions_hidden_states.dtype))
        point_logits = self.depth_point_score_head(depth_tokens + query_seed.unsqueeze(1)).squeeze(-1).float()
        point_features = depth_point_features.to(device=actions_hidden_states.device, dtype=torch.float32)
        valid = point_features[..., 8] > 0.5
        valid = torch.where(valid.any(dim=1, keepdim=True), valid, torch.ones_like(valid))
        point_logits = point_logits.masked_fill(~valid, -1e4)
        point_probs = torch.softmax(point_logits, dim=-1)
        selected_xyz = (point_probs.unsqueeze(-1) * point_features[..., :3]).sum(dim=1)
        selected_rel_xyz = (point_probs.unsqueeze(-1) * point_features[..., 3:6]).sum(dim=1)
        self.last_depth_point_logits = point_logits.detach()
        self.last_depth_point_xyz = selected_xyz.detach()
        return selected_xyz.to(dtype=actions_hidden_states.dtype), selected_rel_xyz.to(dtype=actions_hidden_states.dtype), point_logits

    def predict_point_action_residual(
        self,
        actions_hidden_states,
        depth_context=None,
        depth_point_features=None,
        query_context=None,
    ):
        if (
            not self.use_depth_conditioning
            or self.depth_fusion_type != "object_query"
            or self.depth_point_action_residual is None
            or self.depth_point_action_weight <= 0
            or depth_context is None
            or depth_point_features is None
        ):
            return None
        batch_size = actions_hidden_states.shape[0]
        _, selected_rel_xyz, _ = self.predict_point_keypose_xyz(
            actions_hidden_states,
            depth_context=depth_context,
            depth_point_features=depth_point_features,
            query_context=query_context,
        )
        first_action_context = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM, -1).mean(dim=2)[:, 0]
        adapter_input = torch.cat([first_action_context, selected_rel_xyz], dim=-1)
        gate = self.depth_fusion_gate.to(dtype=actions_hidden_states.dtype)
        residual_xyz = gate * self.depth_point_action_weight * self.depth_point_action_residual(adapter_input)
        residual_clip = self.depth_point_action_clip or self.depth_action_residual_clip
        if residual_clip > 0:
            residual_xyz = residual_xyz.clamp(-residual_clip, residual_clip)
        residual = torch.zeros(
            batch_size,
            NUM_ACTIONS_CHUNK,
            self.action_dim,
            device=actions_hidden_states.device,
            dtype=actions_hidden_states.dtype,
        )
        residual[:, 0, :3] = residual_xyz
        return residual

    def predict_waypoint_xyz_action(
        self,
        actions_hidden_states,
        depth_context=None,
        depth_point_features=None,
        query_context=None,
    ):
        if (
            not self.use_depth_conditioning
            or self.depth_fusion_type != "object_query"
            or self.depth_waypoint_action_weight <= 0
            or depth_context is None
            or depth_point_features is None
        ):
            return None
        _, selected_rel_xyz, _ = self.predict_point_keypose_xyz(
            actions_hidden_states,
            depth_context=depth_context,
            depth_point_features=depth_point_features,
            query_context=query_context,
        )
        waypoint_xyz = selected_rel_xyz
        if self.depth_waypoint_action_clip > 0:
            waypoint_xyz = waypoint_xyz.clamp(-self.depth_waypoint_action_clip, self.depth_waypoint_action_clip)
        waypoint_xyz = waypoint_xyz * self.depth_waypoint_action_scale
        self.last_depth_waypoint_xyz_action = waypoint_xyz.detach()
        return waypoint_xyz

    def predict_spatial_delta(
        self,
        depth_context,
        actions_hidden_states=None,
        query_context=None,
        aux_target="none",
        depth_point_features=None,
    ):
        if not self.use_depth_conditioning or self.spatial_head is None:
            raise ValueError("Spatial auxiliary prediction requires use_depth_conditioning=True")
        if self.depth_fusion_type == "object_query":
            if actions_hidden_states is None:
                raise ValueError("object_query spatial auxiliary prediction requires actions_hidden_states")
            if str(aux_target or "none") in (
                "point_keypose_xyz",
                "visible_object_point_xyz",
                "first_close_pose_xyz",
                "pre_first_close_pose_xyz",
                "visible_first_close_point_xyz",
                "visible_pre_first_close_point_xyz",
                "future_pose_xyz",
                "final_pose_xyz",
                "farthest_future_pose_xyz",
            ):
                selected_xyz, _, _ = self.predict_point_keypose_xyz(
                    actions_hidden_states,
                    depth_context=depth_context,
                    depth_point_features=depth_point_features,
                    query_context=query_context,
                )
                return selected_xyz
            attended = self.object_query_attended_tokens(
                actions_hidden_states, depth_context, query_context=query_context
            )
            aux_target = str(aux_target or "none")
            if aux_target in ("relative_xyz", "contact_xyz", "visible_object_rel_xyz", "ee_to_object_xyz"):
                return self.object_query_spatial_heads["ee_to_object_xyz"](attended[:, 0])
            if aux_target == "object_to_target_xyz":
                return self.object_query_spatial_heads["object_to_target_xyz"](attended[:, 1])
            if aux_target == "gripper_to_contact_distance":
                return self.object_query_spatial_heads["gripper_to_contact_distance"](attended[:, 2])
            if aux_target == "task_3d":
                return torch.cat(
                    [
                        self.object_query_spatial_heads["ee_to_object_xyz"](attended[:, 0]),
                        self.object_query_spatial_heads["object_to_target_xyz"](attended[:, 1]),
                        self.object_query_spatial_heads["gripper_to_contact_distance"](attended[:, 2]),
                    ],
                    dim=-1,
                )
            if aux_target == "distance_bin":
                return self.object_query_spatial_heads["distance_bin"](attended[:, 2])
            if aux_target == "relative_z_bin":
                return self.object_query_spatial_heads["relative_z_bin"](attended[:, 0])
            if aux_target in ("absolute_keypose", "rlbench_keypose_action", "projected_keypose_uv"):
                return self.object_query_spatial_heads["absolute_keypose"](attended.reshape(actions_hidden_states.shape[0], -1))
            if aux_target == "projected_keypose_heatmap":
                logits = self.object_query_spatial_heads["absolute_keypose"](
                    attended.reshape(actions_hidden_states.shape[0], -1)
                )
                return self._projected_heatmap_logits(logits)
            depth_context = self.depth_context(attended.reshape(actions_hidden_states.shape[0], -1))
        else:
            depth_context = self._pool_depth_context(depth_context)
        ref_param = next(self.spatial_head.parameters())
        depth_context = depth_context.to(device=ref_param.device, dtype=ref_param.dtype)
        return self.spatial_head(depth_context)

    def predict_action(self, actions_hidden_states, depth_context=None, query_context=None, depth_point_features=None):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        conditioned_hidden_states = self.condition_action_hidden_states(
            actions_hidden_states, depth_context, query_context=query_context
        )
        rearranged_actions_hidden_states = conditioned_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        residual = self.predict_action_residual(actions_hidden_states, depth_context)
        if residual is not None:
            action = action + residual
        keypose_residual = self.predict_keypose_action_residual(
            actions_hidden_states, depth_context, query_context=query_context
        )
        if keypose_residual is not None:
            action = action + keypose_residual
        point_action_residual = self.predict_point_action_residual(
            actions_hidden_states,
            depth_context=depth_context,
            depth_point_features=depth_point_features,
            query_context=query_context,
        )
        if point_action_residual is not None:
            action = action + point_action_residual
        waypoint_xyz = self.predict_waypoint_xyz_action(
            actions_hidden_states,
            depth_context=depth_context,
            depth_point_features=depth_point_features,
            query_context=query_context,
        )
        if waypoint_xyz is not None:
            weight = max(0.0, min(1.0, self.depth_waypoint_action_weight))
            chunk_len = max(1, min(NUM_ACTIONS_CHUNK, self.depth_waypoint_action_chunk_len))
            waypoint_chunk = waypoint_xyz.to(action.dtype).unsqueeze(1).expand(-1, chunk_len, -1)
            self.last_depth_waypoint_chunk_xyz_action = waypoint_chunk.detach()
            action = action.clone()
            action[:, :chunk_len, :3] = (1.0 - weight) * action[:, :chunk_len, :3] + weight * waypoint_chunk
        return action


class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_diffusion_steps_train=50,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps_train, beta_schedule="squaredcos_cap_v2")
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred
