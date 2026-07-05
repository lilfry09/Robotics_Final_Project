"""
Heatmap Generator for BridgeVLA-style 3D-to-2D spatial guidance.

Projects 3D task-relevant points (object, target, contact) into camera views
and generates 2D Gaussian heatmaps to guide the VLA policy.

Key difference from current DepthVLA approach:
- Creates explicit 2D spatial bottleneck that CANNOT be satisfied with null/shuffled depth
- Heatmaps are interpretable and debuggable
- Matches VLM's 2D image priors better than abstract depth tokens
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


def project_3d_to_2d(
    point_3d: np.ndarray,
    K: np.ndarray,
    T_cam_to_base: np.ndarray
) -> Tuple[float, float, bool]:
    """
    Project a 3D point in base frame to 2D pixel coordinates.

    Args:
        point_3d: (3,) array in base frame coordinates
        K: (3, 3) camera intrinsics matrix
        T_cam_to_base: (4, 4) camera-to-base transform

    Returns:
        u, v: pixel coordinates (float)
        valid: whether projection is valid (in front of camera)
    """
    point_3d = np.asarray(point_3d, dtype=np.float32).reshape(3)

    # Transform from base frame to camera frame
    T_base_to_cam = np.linalg.inv(T_cam_to_base.astype(np.float64)).astype(np.float32)
    point_3d_homo = np.append(point_3d, 1.0)
    point_cam = (T_base_to_cam @ point_3d_homo)[:3]

    # Check if point is in front of camera
    if point_cam[2] <= 0.01:
        return 0.0, 0.0, False

    # Project to image plane
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    u = fx * point_cam[0] / point_cam[2] + cx
    v = fy * point_cam[1] / point_cam[2] + cy

    return float(u), float(v), True


def generate_gaussian_heatmap(
    image_size: Tuple[int, int],
    center_u: float,
    center_v: float,
    sigma: float = 5.0,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate a 2D Gaussian heatmap centered at (center_u, center_v).

    Args:
        image_size: (height, width)
        center_u, center_v: center in pixel coordinates
        sigma: Gaussian standard deviation in pixels
        normalize: whether to normalize to [0, 1]

    Returns:
        heatmap: (height, width) float32 array
    """
    height, width = image_size

    # Create coordinate grids
    y = np.arange(height, dtype=np.float32)
    x = np.arange(width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Compute Gaussian
    heatmap = np.exp(-((xx - center_u)**2 + (yy - center_v)**2) / (2 * sigma**2))

    if normalize:
        max_val = heatmap.max()
        if max_val > 1e-8:
            heatmap = heatmap / max_val

    return heatmap.astype(np.float32)


def generate_task_heatmaps(
    obs_dict: Dict,
    image_size: Tuple[int, int] = (224, 224),
    sigma: float = 5.0,
    view: str = "agentview"
) -> np.ndarray:
    """
    Generate multi-channel task-relevant heatmaps.

    Args:
        obs_dict: observation dictionary with keys:
            - 'manipulated_object_pos': (3,) object position in base frame
            - 'target_pos': (3,) target position in base frame
            - 'ee_pos': (3,) end-effector position
            - f'{view}_K': (3, 3) camera intrinsics
            - f'{view}_T_camera_to_base': (4, 4) camera transform
        image_size: output heatmap size (height, width)
        sigma: Gaussian sigma in pixels
        view: camera view name

    Returns:
        heatmaps: (height, width, 3) float32 array
            Channel 0: object location heatmap
            Channel 1: target location heatmap
            Channel 2: end-effector/contact heatmap
    """
    height, width = image_size

    # Extract required data
    K = obs_dict[f'{view}_K']
    T_cam_to_base = obs_dict[f'{view}_T_camera_to_base']

    # Initialize channels
    heatmap_object = np.zeros((height, width), dtype=np.float32)
    heatmap_target = np.zeros((height, width), dtype=np.float32)
    heatmap_ee = np.zeros((height, width), dtype=np.float32)

    # Channel 0: Object location
    if 'manipulated_object_pos' in obs_dict and obs_dict['manipulated_object_pos'] is not None:
        obj_pos = obs_dict['manipulated_object_pos']
        if np.isfinite(obj_pos).all():
            u_obj, v_obj, valid_obj = project_3d_to_2d(obj_pos, K, T_cam_to_base)
            if valid_obj and 0 <= u_obj < width and 0 <= v_obj < height:
                heatmap_object = generate_gaussian_heatmap(
                    image_size, u_obj, v_obj, sigma, normalize=True
                )

    # Channel 1: Target location
    if 'target_pos' in obs_dict and obs_dict['target_pos'] is not None:
        target_pos = obs_dict['target_pos']
        if np.isfinite(target_pos).all():
            u_target, v_target, valid_target = project_3d_to_2d(target_pos, K, T_cam_to_base)
            if valid_target and 0 <= u_target < width and 0 <= v_target < height:
                heatmap_target = generate_gaussian_heatmap(
                    image_size, u_target, v_target, sigma, normalize=True
                )

    # Channel 2: End-effector location
    if 'ee_pos' in obs_dict and obs_dict['ee_pos'] is not None:
        ee_pos = obs_dict['ee_pos']
        if np.isfinite(ee_pos).all():
            u_ee, v_ee, valid_ee = project_3d_to_2d(ee_pos, K, T_cam_to_base)
            if valid_ee and 0 <= u_ee < width and 0 <= v_ee < height:
                heatmap_ee = generate_gaussian_heatmap(
                    image_size, u_ee, v_ee, sigma, normalize=True
                )

    # Stack channels
    heatmaps = np.stack([heatmap_object, heatmap_target, heatmap_ee], axis=-1)

    return heatmaps


def corrupt_heatmap(heatmap: np.ndarray, mode: str = "shuffle") -> np.ndarray:
    """
    Generate corrupted heatmap for ablation studies.

    Args:
        heatmap: (H, W, C) heatmap array
        mode: corruption mode - "shuffle" or "random"

    Returns:
        corrupted: (H, W, C) corrupted heatmap
    """
    if mode == "shuffle":
        # Shuffle spatial locations independently per channel
        corrupted = np.zeros_like(heatmap)
        for c in range(heatmap.shape[-1]):
            flat = heatmap[..., c].flatten()
            np.random.shuffle(flat)
            corrupted[..., c] = flat.reshape(heatmap.shape[:2])
        return corrupted

    elif mode == "random":
        # Generate random Gaussian heatmaps
        height, width, channels = heatmap.shape
        corrupted = np.zeros_like(heatmap)
        for c in range(channels):
            rand_u = np.random.uniform(0, width)
            rand_v = np.random.uniform(0, height)
            corrupted[..., c] = generate_gaussian_heatmap(
                (height, width), rand_u, rand_v, sigma=5.0, normalize=True
            )
        return corrupted

    else:
        raise ValueError(f"Unknown corruption mode: {mode}")


# Torch utilities for training
def heatmap_to_tensor(heatmap: np.ndarray) -> torch.Tensor:
    """Convert numpy heatmap (H, W, C) to torch tensor (C, H, W)."""
    return torch.from_numpy(heatmap).permute(2, 0, 1).float()


def tensor_to_heatmap(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor (C, H, W) to numpy heatmap (H, W, C)."""
    return tensor.permute(1, 2, 0).cpu().numpy()


# ============================================================================
# Batch Processing for Training
# ============================================================================

def batch_project_3d_to_2d(
    points_3d_base: torch.Tensor,
    camera_K: torch.Tensor,
    camera_T_base: torch.Tensor,
    image_height: int = 256,
    image_width: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch version: Project 3D points in base frame to 2D image coordinates.

    Args:
        points_3d_base: (B, N, 3) 3D points in base/world frame [X, Y, Z]
        camera_K: (B, 3, 3) or (3, 3) camera intrinsics
        camera_T_base: (B, 4, 4) or (4, 4) camera-to-base transform [R|t]
        image_height: image height in pixels
        image_width: image width in pixels

    Returns:
        uv: (B, N, 2) pixel coordinates [u, v] (width, height)
        valid_mask: (B, N) bool mask for points in image bounds and positive depth
    """
    B, N, _ = points_3d_base.shape
    device = points_3d_base.device

    # Expand camera params if needed
    if camera_K.dim() == 2:
        camera_K = camera_K.unsqueeze(0).expand(B, -1, -1)
    if camera_T_base.dim() == 2:
        camera_T_base = camera_T_base.unsqueeze(0).expand(B, -1, -1)

    # 1. Transform from base frame to camera frame
    T_base_to_camera = torch.inverse(camera_T_base)  # (B, 4, 4)

    # Convert to homogeneous coordinates
    points_3d_homo = torch.cat([
        points_3d_base,
        torch.ones(B, N, 1, device=device)
    ], dim=-1)  # (B, N, 4)

    # Batch matrix multiply: (B, 4, 4) @ (B, 4, N) -> (B, 4, N)
    points_3d_camera = torch.bmm(T_base_to_camera, points_3d_homo.transpose(1, 2))
    points_3d_camera = points_3d_camera.transpose(1, 2)[..., :3]  # (B, N, 3)

    # 2. Project to image plane using intrinsics
    # (B, 3, 3) @ (B, 3, N) -> (B, 3, N)
    points_2d_homo = torch.bmm(camera_K, points_3d_camera.transpose(1, 2))
    points_2d_homo = points_2d_homo.transpose(1, 2)  # (B, N, 3)

    # 3. Normalize by depth (perspective division)
    z = points_2d_homo[..., 2:3]  # (B, N, 1) depth
    uv = points_2d_homo[..., :2] / (z + 1e-8)  # (B, N, 2) [u, v]

    # 4. Compute valid mask
    valid_mask = (
        (uv[..., 0] >= 0) & (uv[..., 0] < image_width) &
        (uv[..., 1] >= 0) & (uv[..., 1] < image_height) &
        (z[..., 0] > 0)
    )

    return uv, valid_mask


def batch_generate_gaussian_heatmap(
    centers_uv: torch.Tensor,
    valid_mask: torch.Tensor,
    image_height: int = 256,
    image_width: int = 256,
    sigma: float = 15.0,
) -> torch.Tensor:
    """
    Batch generate Gaussian heatmaps.

    Args:
        centers_uv: (B, 2) pixel coordinates [u, v]
        valid_mask: (B,) bool mask
        image_height: heatmap height
        image_width: heatmap width
        sigma: Gaussian std

    Returns:
        heatmaps: (B, H, W) normalized to [0, 1]
    """
    B = centers_uv.shape[0]
    device = centers_uv.device

    # Create coordinate grids (shared across batch)
    y = torch.arange(image_height, device=device, dtype=torch.float32)
    x = torch.arange(image_width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # (H, W)
    yy = yy.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    xx = xx.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)

    # Compute Gaussian for all batch elements
    u_centers = centers_uv[:, 0].view(B, 1, 1)  # (B, 1, 1)
    v_centers = centers_uv[:, 1].view(B, 1, 1)  # (B, 1, 1)

    heatmaps = torch.exp(
        -((xx - u_centers)**2 + (yy - v_centers)**2) / (2 * sigma**2)
    )  # (B, H, W)

    # Normalize each heatmap individually
    max_vals = heatmaps.view(B, -1).max(dim=1, keepdim=True)[0]  # (B, 1)
    max_vals = max_vals.view(B, 1, 1)  # (B, 1, 1)
    heatmaps = heatmaps / (max_vals + 1e-8)

    # Zero out invalid heatmaps
    valid_mask = valid_mask.view(B, 1, 1).float()
    heatmaps = heatmaps * valid_mask

    return heatmaps  # (B, H, W)


def create_heatmap_labels_batch(
    manipulated_object_pos: torch.Tensor,
    target_pos: torch.Tensor,
    camera_K: torch.Tensor,
    camera_T_base: torch.Tensor,
    image_height: int = 256,
    image_width: int = 256,
    sigma: float = 15.0,
) -> Dict[str, torch.Tensor]:
    """
    Create object and target heatmap labels for a batch.

    Args:
        manipulated_object_pos: (B, 3) object positions in base frame
        target_pos: (B, 3) target positions in base frame
        camera_K: (B, 3, 3) or (3, 3) camera intrinsics
        camera_T_base: (B, 4, 4) or (4, 4) camera-to-base transform
        image_height: output heatmap height
        image_width: output heatmap width
        sigma: Gaussian std for heatmap

    Returns:
        Dict containing:
            'object_heatmap': (B, H, W) object location heatmaps
            'target_heatmap': (B, H, W) target location heatmaps
            'object_valid': (B,) bool mask
            'target_valid': (B,) bool mask
    """
    B = manipulated_object_pos.shape[0]

    # Project object positions to 2D
    object_uv, object_valid = batch_project_3d_to_2d(
        manipulated_object_pos.unsqueeze(1),  # (B, 1, 3)
        camera_K,
        camera_T_base,
        image_height,
        image_width
    )
    object_uv = object_uv.squeeze(1)  # (B, 2)
    object_valid = object_valid.squeeze(1)  # (B,)

    # Project target positions to 2D
    target_uv, target_valid = batch_project_3d_to_2d(
        target_pos.unsqueeze(1),  # (B, 1, 3)
        camera_K,
        camera_T_base,
        image_height,
        image_width
    )
    target_uv = target_uv.squeeze(1)  # (B, 2)
    target_valid = target_valid.squeeze(1)  # (B,)

    # Generate heatmaps
    object_heatmap = batch_generate_gaussian_heatmap(
        object_uv, object_valid, image_height, image_width, sigma
    )
    target_heatmap = batch_generate_gaussian_heatmap(
        target_uv, target_valid, image_height, image_width, sigma
    )

    return {
        'object_heatmap': object_heatmap,  # (B, H, W)
        'target_heatmap': target_heatmap,  # (B, H, W)
        'object_valid': object_valid,       # (B,)
        'target_valid': target_valid,       # (B,)
    }


# ============================================================================
# Loss Functions
# ============================================================================

def compute_heatmap_mse_loss(
    pred_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    valid_mask: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute MSE loss for heatmap prediction.

    Args:
        pred_heatmap: (B, H, W) predicted heatmap
        gt_heatmap: (B, H, W) ground truth heatmap
        valid_mask: (B,) bool mask (True if object is visible)
        reduction: 'mean' or 'sum'

    Returns:
        loss: scalar
    """
    # MSE per sample
    mse = F.mse_loss(pred_heatmap, gt_heatmap, reduction='none')  # (B, H, W)
    mse = mse.mean(dim=[1, 2])  # (B,)

    # Weight by validity
    valid_mask = valid_mask.float()
    if reduction == 'mean':
        loss = (mse * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    elif reduction == 'sum':
        loss = (mse * valid_mask).sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return loss


def compute_heatmap_focal_loss(
    pred_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Focal loss for heatmap (encourages sharper peaks).

    Based on CornerNet focal loss.

    Args:
        pred_heatmap: (B, H, W) predicted heatmap in [0, 1]
        gt_heatmap: (B, H, W) ground truth heatmap in [0, 1]
        valid_mask: (B,) bool mask
        alpha: penalty weight for positive samples
        beta: penalty weight for negative samples
        reduction: 'mean' or 'sum'

    Returns:
        loss: scalar
    """
    pred = pred_heatmap.clamp(min=1e-6, max=1 - 1e-6)
    gt = gt_heatmap

    # Positive locations (gt > 0)
    pos_mask = (gt >= 0.99).float()
    pos_loss = -torch.pow(1 - pred, alpha) * torch.log(pred) * pos_mask

    # Negative locations (gt < 1)
    neg_mask = (gt < 0.99).float()
    neg_loss = -torch.pow(1 - gt, beta) * torch.pow(pred, alpha) * torch.log(1 - pred) * neg_mask

    # Combine
    focal_loss = (pos_loss + neg_loss).sum(dim=[1, 2])  # (B,)

    # Weight by validity
    valid_mask = valid_mask.float()
    if reduction == 'mean':
        loss = (focal_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    elif reduction == 'sum':
        loss = (focal_loss * valid_mask).sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return loss


def compute_heatmap_metrics(
    pred_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for heatmap prediction.

    Args:
        pred_heatmap: (B, H, W) predicted heatmap
        gt_heatmap: (B, H, W) ground truth heatmap
        valid_mask: (B,) bool mask

    Returns:
        Dict of metrics: mse, psnr, peak_distance, etc.
    """
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return {
            'mse': 0.0,
            'psnr': 0.0,
            'peak_distance': 0.0,
            'num_valid': 0
        }

    pred = pred_heatmap[valid_indices]
    gt = gt_heatmap[valid_indices]

    # MSE
    mse = F.mse_loss(pred, gt).item()

    # PSNR
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))

    # Peak localization error
    pred_flat = pred.view(len(valid_indices), -1)
    gt_flat = gt.view(len(valid_indices), -1)

    pred_peaks = torch.argmax(pred_flat, dim=1)  # (N,)
    gt_peaks = torch.argmax(gt_flat, dim=1)      # (N,)

    # Convert flat indices to 2D coordinates
    H, W = pred.shape[1], pred.shape[2]
    pred_y = pred_peaks // W
    pred_x = pred_peaks % W
    gt_y = gt_peaks // W
    gt_x = gt_peaks % W

    # Euclidean distance in pixels
    peak_dist = torch.sqrt(
        (pred_x - gt_x).float()**2 + (pred_y - gt_y).float()**2
    ).mean().item()

    return {
        'mse': mse,
        'psnr': psnr,
        'peak_distance': peak_dist,
        'num_valid': len(valid_indices)
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_heatmap_overlay(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet',
) -> np.ndarray:
    """
    Overlay heatmap on RGB image for visualization.

    Args:
        rgb_image: (H, W, 3) uint8 RGB image
        heatmap: (H, W) float heatmap [0, 1]
        alpha: blending factor
        colormap: matplotlib colormap name

    Returns:
        overlay: (H, W, 3) uint8 blended image
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # Normalize heatmap
    heatmap = np.clip(heatmap, 0, 1)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

    # Ensure RGB image is uint8
    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8) if rgb_image.max() <= 1.0 else rgb_image.astype(np.uint8)

    # Blend
    overlay = (alpha * heatmap_colored + (1 - alpha) * rgb_image).astype(np.uint8)

    return overlay


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("Testing heatmap_generator.py")
    print("=" * 60)

    # Test 1: Single projection
    print("\n[Test 1] Single 3D→2D projection...")
    point_3d = np.array([0.0, 0.0, 1.0])  # Point 1m in front of camera
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = K[1, 1] = 200.0  # focal length
    K[0, 2] = K[1, 2] = 128.0  # principal point
    T_cam_base = np.eye(4, dtype=np.float32)  # Camera at origin looking along +Z

    u, v, valid = project_3d_to_2d(point_3d, K, T_cam_base)
    print(f"  Point 3D: {point_3d}")
    print(f"  Point 2D: ({u:.1f}, {v:.1f}), valid={valid}")
    assert valid, "Projection should be valid"
    assert abs(u - 128.0) < 1.0 and abs(v - 128.0) < 1.0, "Should project to image center"

    # Test 2: Heatmap generation
    print("\n[Test 2] Single heatmap generation...")
    hm = generate_gaussian_heatmap((256, 256), 128.0, 128.0, sigma=15.0)
    print(f"  Heatmap shape: {hm.shape}")
    print(f"  Heatmap range: [{hm.min():.4f}, {hm.max():.4f}]")
    print(f"  Center value: {hm[128, 128]:.4f}")
    assert hm.max() > 0.99, "Peak should be normalized to ~1.0"

    # Test 3: Batch projection
    print("\n[Test 3] Batch 3D→2D projection...")
    B = 4
    points_3d_batch = torch.randn(B, 1, 3) * 0.2 + torch.tensor([[[0.0, 0.0, 1.0]]])
    K_batch = torch.from_numpy(K).unsqueeze(0).expand(B, -1, -1)
    T_batch = torch.from_numpy(T_cam_base).unsqueeze(0).expand(B, -1, -1)

    uv_batch, valid_batch = batch_project_3d_to_2d(points_3d_batch, K_batch, T_batch, 256, 256)
    print(f"  Batch UV shape: {uv_batch.shape}")
    print(f"  Valid count: {valid_batch.sum().item()}/{B}")
    assert valid_batch.sum().item() >= B - 1, "Most projections should be valid"

    # Test 4: Batch heatmap generation
    print("\n[Test 4] Batch heatmap generation...")
    centers = uv_batch.squeeze(1)  # (B, 2)
    valid = valid_batch.squeeze(1)  # (B,)
    hm_batch = batch_generate_gaussian_heatmap(centers, valid, 256, 256, sigma=15.0)
    print(f"  Batch heatmap shape: {hm_batch.shape}")
    print(f"  Valid heatmaps: {(hm_batch.max(dim=1)[0].max(dim=1)[0] > 0.5).sum().item()}/{B}")

    # Test 5: Loss computation
    print("\n[Test 5] Heatmap loss...")
    pred = hm_batch + torch.randn_like(hm_batch) * 0.1
    pred = pred.clamp(0, 1)
    loss = compute_heatmap_mse_loss(pred, hm_batch, valid)
    print(f"  MSE loss: {loss.item():.6f}")

    # Test 6: Metrics
    print("\n[Test 6] Heatmap metrics...")
    metrics = compute_heatmap_metrics(pred, hm_batch, valid)
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  Peak distance: {metrics['peak_distance']:.2f} pixels")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
