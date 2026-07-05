"""
DepthVLA Advanced Training Components
Implements curriculum dropout, multi-level contrastive, and hierarchical supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Component 1: Curriculum Depth Dropout
# ============================================================================

class CurriculumDepthDropout(nn.Module):
    """
    Gradually decrease dropout rate during training
    Forces model to work without depth early, then learn to leverage it
    """

    def __init__(self, initial_rate=0.5, final_rate=0.2, total_steps=10000, schedule='cosine'):
        super().__init__()
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.total_steps = total_steps
        self.schedule = schedule
        self.current_step = 0

    def get_dropout_rate(self):
        progress = min(1.0, self.current_step / self.total_steps)

        if self.schedule == 'linear':
            rate = self.initial_rate + (self.final_rate - self.initial_rate) * progress
        elif self.schedule == 'cosine':
            rate = self.final_rate + (self.initial_rate - self.final_rate) * \
                   0.5 * (1 + np.cos(np.pi * progress))
        elif self.schedule == 'exponential':
            rate = self.initial_rate * (self.final_rate / self.initial_rate) ** progress
        else:
            rate = self.initial_rate

        return rate

    def forward(self, depth_features, training=True):
        if not training:
            return depth_features

        dropout_rate = self.get_dropout_rate()
        batch_size = depth_features.shape[0]

        # Per-sample dropout
        keep_mask = torch.rand(batch_size, device=depth_features.device) > dropout_rate
        keep_mask = keep_mask.float().view(-1, 1, 1)

        return depth_features * keep_mask

    def step(self):
        self.current_step += 1
        return self.get_dropout_rate()


# ============================================================================
# Component 2: Multi-Level Contrastive Loss
# ============================================================================

class MultiLevelContrastiveLoss(nn.Module):
    """
    Enforce depth usage at action, representation, and residual levels
    """

    def __init__(self, margin=0.05, hierarchy_weights=None):
        super().__init__()
        self.margin = margin
        self.hierarchy_weights = hierarchy_weights or [1.0, 0.5, 0.3]

    def forward(self, outputs_normal, outputs_null, labels):
        """
        Args:
            outputs_normal: dict with keys ['actions', 'hidden', 'residual']
            outputs_null: dict with keys ['actions', 'hidden', 'residual']
            labels: ground truth actions
        """
        losses = []

        # Level 1: Action prediction loss
        loss_action_normal = F.smooth_l1_loss(outputs_normal['actions'], labels)
        loss_action_null = F.smooth_l1_loss(outputs_null['actions'], labels)
        loss_action_ranking = F.relu(loss_action_normal - loss_action_null + self.margin)
        losses.append(loss_action_ranking)

        # Level 2: Representation diversity
        if 'hidden' in outputs_normal and 'hidden' in outputs_null:
            hidden_normal = outputs_normal['hidden']
            hidden_null = outputs_null['hidden']

            # Cosine similarity
            cos_sim = F.cosine_similarity(
                hidden_normal.view(hidden_normal.size(0), -1),
                hidden_null.view(hidden_null.size(0), -1),
                dim=1
            ).mean()

            # Encourage different representations (sim < 0.8)
            loss_diversity = F.relu(cos_sim - 0.8)
            losses.append(loss_diversity)
        else:
            losses.append(torch.tensor(0.0, device=labels.device))

        # Level 3: Residual magnitude
        if 'residual' in outputs_normal and 'residual' in outputs_null:
            residual_normal = outputs_normal['residual']
            residual_null = outputs_null['residual']

            # Residuals should be different
            residual_diff = F.mse_loss(residual_normal, residual_null)
            loss_residual = F.relu(0.01 - residual_diff)  # At least 1% difference
            losses.append(loss_residual)
        else:
            losses.append(torch.tensor(0.0, device=labels.device))

        # Weighted sum
        total_loss = sum(w * l for w, l in zip(self.hierarchy_weights, losses))

        return total_loss, {
            'contrastive_action': losses[0].item(),
            'contrastive_diversity': losses[1].item(),
            'contrastive_residual': losses[2].item(),
        }


# ============================================================================
# Component 3: Hierarchical Spatial Supervision
# ============================================================================

class HierarchicalSpatialLoss(nn.Module):
    """
    Multi-scale 3D supervision: coarse workspace → medium object-vectors → fine contacts
    """

    def __init__(self, coarse_weight=0.3, medium_weight=0.5, fine_weight=0.2):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.medium_weight = medium_weight
        self.fine_weight = fine_weight

        # Prediction heads
        self.coarse_head = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 8*8*8),  # 8x8x8 occupancy grid
            nn.Sigmoid()
        )

        self.medium_head = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 9)  # ee_to_obj(3) + obj_to_target(3) + ee_to_target(3)
        )

        self.fine_head = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 7)  # contact_point(3) + contact_normal(3) + grasp_quality(1)
        )

    def forward(self, depth_context, targets):
        """
        Args:
            depth_context: [B, 4096] global depth summary
            targets: dict with keys ['occupancy', 'vectors', 'contact']
        """
        losses = {}

        # Coarse: workspace occupancy
        if 'occupancy' in targets:
            pred_occupancy = self.coarse_head(depth_context)
            gt_occupancy = targets['occupancy'].view(targets['occupancy'].size(0), -1)
            losses['coarse'] = F.binary_cross_entropy(pred_occupancy, gt_occupancy)
        else:
            losses['coarse'] = torch.tensor(0.0, device=depth_context.device)

        # Medium: object-relative vectors
        if 'vectors' in targets:
            pred_vectors = self.medium_head(depth_context)
            losses['medium'] = F.smooth_l1_loss(pred_vectors, targets['vectors'])
        else:
            losses['medium'] = torch.tensor(0.0, device=depth_context.device)

        # Fine: contact prediction
        if 'contact' in targets:
            pred_contact = self.fine_head(depth_context)
            losses['fine'] = F.smooth_l1_loss(pred_contact, targets['contact'])
        else:
            losses['fine'] = torch.tensor(0.0, device=depth_context.device)

        # Total
        total_loss = (self.coarse_weight * losses['coarse'] +
                     self.medium_weight * losses['medium'] +
                     self.fine_weight * losses['fine'])

        return total_loss, {k: v.item() for k, v in losses.items()}


# ============================================================================
# Component 4: Integrated Training Loop
# ============================================================================

class DepthVLAAdvancedTrainer:
    """
    Training loop with all advanced components
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Components
        self.depth_dropout = CurriculumDepthDropout(
            initial_rate=config.get('depth_dropout_initial', 0.5),
            final_rate=config.get('depth_dropout_final', 0.2),
            total_steps=config.get('depth_dropout_steps', 10000),
            schedule=config.get('depth_dropout_schedule', 'cosine')
        )

        self.contrastive_loss = MultiLevelContrastiveLoss(
            margin=config.get('contrastive_margin', 0.05),
            hierarchy_weights=config.get('contrastive_hierarchy_weights', [1.0, 0.5, 0.3])
        )

        self.spatial_loss = HierarchicalSpatialLoss(
            coarse_weight=config.get('spatial_coarse_weight', 0.3),
            medium_weight=config.get('spatial_medium_weight', 0.5),
            fine_weight=config.get('spatial_fine_weight', 0.2)
        )

        self.contrastive_weight = config.get('contrastive_weight', 0.3)
        self.spatial_weight = config.get('spatial_weight', 0.2)

    def compute_loss(self, batch):
        """
        Compute full training loss with all components
        """
        rgb = batch['rgb']
        depth = batch['depth']
        proprio = batch['proprio']
        labels = batch['actions']

        # Forward with normal depth (with dropout)
        depth_features = self.model.depth_encoder(
            depth, batch['K'], batch['T'], proprio[:, :3]
        )
        depth_features = self.depth_dropout(depth_features, training=True)

        outputs_normal = self.model.forward_with_outputs(
            rgb, depth_features, proprio
        )

        # Forward with null depth (for contrastive)
        depth_features_null = torch.zeros_like(depth_features)
        outputs_null = self.model.forward_with_outputs(
            rgb, depth_features_null, proprio
        )

        # Loss 1: Main action BC loss
        loss_action = F.smooth_l1_loss(outputs_normal['actions'], labels)

        # Loss 2: Multi-level contrastive
        loss_contrastive, contrastive_metrics = self.contrastive_loss(
            outputs_normal, outputs_null, labels
        )

        # Loss 3: Hierarchical spatial supervision
        if 'spatial_targets' in batch:
            depth_context = outputs_normal.get('depth_context', depth_features.mean(dim=1))
            loss_spatial, spatial_metrics = self.spatial_loss(
                depth_context, batch['spatial_targets']
            )
        else:
            loss_spatial = torch.tensor(0.0, device=labels.device)
            spatial_metrics = {}

        # Total loss
        total_loss = (loss_action +
                     self.contrastive_weight * loss_contrastive +
                     self.spatial_weight * loss_spatial)

        # Step dropout scheduler
        current_dropout_rate = self.depth_dropout.step()

        # Metrics
        metrics = {
            'loss/total': total_loss.item(),
            'loss/action': loss_action.item(),
            'loss/contrastive': loss_contrastive.item(),
            'loss/spatial': loss_spatial.item(),
            'depth/dropout_rate': current_dropout_rate,
            **{f'contrastive/{k}': v for k, v in contrastive_metrics.items()},
            **{f'spatial/{k}': v for k, v in spatial_metrics.items()},
        }

        return total_loss, metrics


# ============================================================================
# Component 5: Spatial Target Computation
# ============================================================================

def compute_spatial_targets(depth, K, T, ee_pos, object_masks=None, target_masks=None):
    """
    Compute hierarchical spatial targets from depth

    Args:
        depth: [B, H, W] metric depth
        K: [B, 3, 3] camera intrinsics
        T: [B, 4, 4] camera extrinsics (cam to base)
        ee_pos: [B, 3] end-effector position in base frame
        object_masks: [B, H, W] object segmentation (optional)
        target_masks: [B, H, W] target location masks (optional)

    Returns:
        dict with spatial targets
    """
    batch_size = depth.shape[0]
    device = depth.device
    targets = {}

    # Backproject depth to 3D points in base frame
    points_base = backproject_depth_to_base(depth, K, T)  # [B, H*W, 3]

    # Level 1: Coarse workspace occupancy (8x8x8 voxel grid)
    occupancy = voxelize_points(points_base, resolution=8, workspace_bounds=[
        [-0.5, 0.5],  # x
        [-0.5, 0.5],  # y
        [0.0, 1.0]    # z
    ])  # [B, 8, 8, 8]
    targets['occupancy'] = occupancy

    # Level 2: Medium - object-relative vectors
    if object_masks is not None and target_masks is not None:
        # Compute object and target centers from masks
        object_centers = compute_masked_centroid(points_base, object_masks)  # [B, 3]
        target_centers = compute_masked_centroid(points_base, target_masks)  # [B, 3]

        ee_to_obj = object_centers - ee_pos
        obj_to_target = target_centers - object_centers
        ee_to_target = target_centers - ee_pos

        targets['vectors'] = torch.cat([ee_to_obj, obj_to_target, ee_to_target], dim=1)  # [B, 9]
    else:
        # Fallback: use nearest point as proxy
        nearest_points = find_nearest_points(points_base, ee_pos)  # [B, 3]
        ee_to_nearest = nearest_points - ee_pos
        targets['vectors'] = torch.cat([ee_to_nearest, ee_to_nearest, ee_to_nearest], dim=1)

    # Level 3: Fine - contact prediction
    contact_points = find_nearest_surface_points(points_base, ee_pos)  # [B, 3]
    contact_normals = estimate_surface_normals(points_base, contact_points)  # [B, 3]
    grasp_quality = estimate_grasp_quality(points_base, ee_pos)  # [B, 1]

    targets['contact'] = torch.cat([contact_points, contact_normals, grasp_quality], dim=1)  # [B, 7]

    return targets


def backproject_depth_to_base(depth, K, T):
    """Backproject depth to 3D points in base frame"""
    B, H, W = depth.shape
    device = depth.device

    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    # Backproject to camera frame
    fx, fy = K[:, 0, 0], K[:, 1, 1]
    cx, cy = K[:, 0, 2], K[:, 1, 2]

    x = x.unsqueeze(0).expand(B, -1, -1).float()
    y = y.unsqueeze(0).expand(B, -1, -1).float()

    z = depth
    x = (x - cx.view(-1, 1, 1)) * z / fx.view(-1, 1, 1)
    y = (y - cy.view(-1, 1, 1)) * z / fy.view(-1, 1, 1)

    points_cam = torch.stack([x, y, z], dim=-1).reshape(B, H*W, 3)

    # Transform to base frame
    points_cam_homo = torch.cat([
        points_cam,
        torch.ones(B, H*W, 1, device=device)
    ], dim=-1)  # [B, H*W, 4]

    points_base_homo = torch.bmm(points_cam_homo, T.transpose(1, 2))  # [B, H*W, 4]
    points_base = points_base_homo[:, :, :3]

    return points_base


def voxelize_points(points, resolution=8, workspace_bounds=None):
    """Convert point cloud to voxel occupancy grid"""
    if workspace_bounds is None:
        workspace_bounds = [[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]]

    B, N, _ = points.shape
    device = points.device

    voxel_grid = torch.zeros(B, resolution, resolution, resolution, device=device)

    for b in range(B):
        pts = points[b]  # [N, 3]

        # Filter points within workspace
        valid_mask = (
            (pts[:, 0] >= workspace_bounds[0][0]) & (pts[:, 0] <= workspace_bounds[0][1]) &
            (pts[:, 1] >= workspace_bounds[1][0]) & (pts[:, 1] <= workspace_bounds[1][1]) &
            (pts[:, 2] >= workspace_bounds[2][0]) & (pts[:, 2] <= workspace_bounds[2][1])
        )
        pts_valid = pts[valid_mask]

        if len(pts_valid) == 0:
            continue

        # Compute voxel indices
        voxel_indices = torch.zeros_like(pts_valid, dtype=torch.long)
        for i, bounds in enumerate(workspace_bounds):
            voxel_indices[:, i] = ((pts_valid[:, i] - bounds[0]) /
                                  (bounds[1] - bounds[0]) * resolution).long()
            voxel_indices[:, i] = torch.clamp(voxel_indices[:, i], 0, resolution - 1)

        # Set occupancy
        voxel_grid[b, voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0

    return voxel_grid


def find_nearest_points(points, query_points):
    """Find nearest point in point cloud for each query"""
    # points: [B, N, 3]
    # query_points: [B, 3]
    distances = torch.norm(points - query_points.unsqueeze(1), dim=2)  # [B, N]
    nearest_idx = distances.argmin(dim=1)  # [B]
    nearest_points = points[torch.arange(points.size(0)), nearest_idx]  # [B, 3]
    return nearest_points


def find_nearest_surface_points(points, ee_pos):
    """Find nearest surface points (for contact prediction)"""
    return find_nearest_points(points, ee_pos)


def estimate_surface_normals(points, query_points, k=20):
    """Estimate surface normals at query points using local neighborhoods"""
    # Simplified: return upward normal
    B = query_points.shape[0]
    device = query_points.device
    normals = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(B, 3)
    return normals


def estimate_grasp_quality(points, ee_pos):
    """Estimate grasp quality based on local geometry"""
    # Simplified: inverse distance to nearest point
    nearest_dist = torch.norm(find_nearest_points(points, ee_pos) - ee_pos, dim=1, keepdim=True)
    quality = 1.0 / (nearest_dist + 0.01)
    return torch.clamp(quality, 0, 1)


def compute_masked_centroid(points, mask):
    """Compute centroid of points within mask"""
    # points: [B, N, 3]
    # mask: [B, H, W] -> flatten to [B, N]
    B = points.shape[0]
    mask_flat = mask.reshape(B, -1).unsqueeze(-1).float()  # [B, N, 1]

    masked_points = points * mask_flat
    centroids = masked_points.sum(dim=1) / (mask_flat.sum(dim=1) + 1e-6)  # [B, 3]

    return centroids


if __name__ == "__main__":
    # Test components
    print("Testing advanced training components...")

    # Test dropout
    dropout = CurriculumDepthDropout(0.5, 0.2, 1000)
    for step in range(0, 1001, 100):
        rate = dropout.get_dropout_rate()
        print(f"Step {step}: dropout rate = {rate:.3f}")
        dropout.step()

    print("\nAll components loaded successfully!")
