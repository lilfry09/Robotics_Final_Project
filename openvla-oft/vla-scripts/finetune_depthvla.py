"""
finetune_depthvla.py

Fine-tunes OpenVLA-OFT on regenerated LIBERO RGB-D HDF5 demonstrations.
Set --use_depth False to train the matched RGB-only baseline on the same data.
"""

import hashlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import draccus
import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from experiments.robot.openvla_utils import check_model_logic_mismatch, model_is_on_hf_hub, update_auto_map
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.depth_encoder import GEOMETRY_CONTINUOUS_FEATURE_NAMES, LightweightDepthTokenEncoder
from prismatic.models.dense_point_depth_encoder import DensePointDepthTokenEncoder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import compute_actions_l1_loss, compute_token_accuracy, get_current_action_mask, get_next_actions_mask
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class DepthFinetuneConfig:
    # Model and data
    vla_path: str = "openvla/openvla-7b"
    rgbd_data_dir: Path = Path("datasets/libero_rgbd_hdf5")
    dataset_name: str = "libero_spatial_rgbd"
    run_root_dir: Path = Path("runs")

    # DepthVLA
    depth_integration_mode: Optional[str] = "depth_object_query"  # rgb_only|depth_prefix_append|depth_action_fusion|depth_action_residual|depth_action_summary_aux|depth_object_query
    use_depth: bool = True
    depth_encoder_type: str = "grid"  # grid|dense_point
    depth_grid_size: int = 4
    depth_num_points_per_view: int = 1024
    depth_hidden_dim: int = 256
    depth_min_m: float = 0.01
    depth_max_m: float = 5.0
    geometry_norm: str = "none"
    geometry_clip: float = 5.0
    depth_fusion_mode: str = "object_query"  # legacy: prefix|action_head|action_residual|action_summary_aux|object_query
    depth_action_fusion_gate_init: float = 0.001
    depth_hidden_delta_clip: float = 0.0
    depth_action_residual_clip: float = 0.0
    depth_keypose_residual_weight: float = 0.0
    depth_keypose_residual_clip: float = 0.0
    depth_point_action_weight: float = 0.0
    depth_point_action_clip: float = 0.0
    depth_waypoint_action_weight: float = 0.0
    depth_waypoint_action_clip: float = 0.0
    depth_waypoint_action_scale: float = 1.0
    depth_waypoint_action_chunk_len: int = 1
    depth_adapter_hidden_dim: int = 256
    summary_repr: str = "base_xyz"  # Active v1 action-summary representation
    summary_pool: str = "meanmax"  # Reserved for old v2 ablations; ignored by v1
    depth_aux_spatial_loss_weight: float = 0.05
    # none|next_action_xyz|relative_xyz|contact_xyz|visible_object_rel_xyz|visible_object_point_xyz|ee_to_object_xyz|object_to_target_xyz|gripper_to_contact_distance|task_3d|absolute_keypose|point_keypose_xyz|first_close_pose_xyz|pre_first_close_pose_xyz|visible_first_close_point_xyz|visible_pre_first_close_point_xyz|future_pose_xyz|final_pose_xyz|farthest_future_pose_xyz|projected_keypose_uv|projected_keypose_heatmap|relative_z_bin|distance_bin
    aux_target: str = "task_3d"
    aux_output_dim: int = 7
    aux_future_horizon: int = 10
    aux_heatmap_size: int = 16
    aux_heatmap_sigma: float = 1.5
    aux_distance_bin_edges: str = "0.036,0.065"
    aux_z_bin_edges: str = "-0.04,0.04"
    freeze_vla_lora: bool = False
    freeze_proprio_projector: bool = False
    freeze_action_head_base: bool = False

    # Depth scale control
    depth_alpha_init: Optional[float] = None
    freeze_depth_alpha: bool = False
    min_depth_alpha: Optional[float] = None

    # Depth causality regularization
    depth_dropout: float = 0.0
    use_contrastive: bool = False
    contrastive_weight: float = 0.0
    contrastive_margin: float = 0.05
    null_to_base_weight: float = 0.0
    corrupt_to_base_weight: float = 0.0
    corrupt_depth_mode: str = "shuffle_depth"

    # OFT settings
    num_images_in_input: int = 2
    use_proprio: bool = True
    batch_size: int = 4
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 100_000
    grad_accumulation_steps: int = 1
    max_steps: int = 150_000
    save_freq: int = 10_000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: Optional[int] = None
    resume_components_from: Optional[str] = None
    image_aug: bool = False

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = True

    # Logging
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10


def remove_ddp_in_checkpoint(state_dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if step is None:
        checkpoint_path = os.path.join(path, f"{module_name}--latest_checkpoint.pt")
    else:
        checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    return remove_ddp_in_checkpoint(torch.load(checkpoint_path, weights_only=True, map_location=device))


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    if not any(param.requires_grad for param in module.parameters()):
        return module
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def encode_depth_context(
    depth_encoder,
    depth_values: torch.Tensor,
    depth_intrinsics: torch.Tensor,
    depth_extrinsics: torch.Tensor,
    depth_valid_mask: torch.Tensor | None = None,
    depth_ee_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(unwrap_module(depth_encoder), DensePointDepthTokenEncoder):
        return depth_encoder(
            depth_values=depth_values,
            depth_intrinsics=depth_intrinsics,
            depth_extrinsics=depth_extrinsics,
            depth_valid_mask=depth_valid_mask,
            ee_pos=depth_ee_pos,
        )
    return depth_encoder(
        depth_values=depth_values,
        depth_intrinsics=depth_intrinsics,
        depth_extrinsics=depth_extrinsics,
        depth_valid_mask=depth_valid_mask,
    )


def compute_depth_point_features(
    depth_encoder,
    depth_values: torch.Tensor,
    depth_intrinsics: torch.Tensor,
    depth_extrinsics: torch.Tensor,
    depth_valid_mask: torch.Tensor | None = None,
    depth_ee_pos: torch.Tensor | None = None,
) -> torch.Tensor | None:
    depth_encoder_module = unwrap_module(depth_encoder)
    if not hasattr(depth_encoder_module, "compute_point_features"):
        return None
    return depth_encoder_module.compute_point_features(
        depth_values=depth_values,
        depth_intrinsics=depth_intrinsics,
        depth_extrinsics=depth_extrinsics,
        depth_valid_mask=depth_valid_mask,
        ee_pos=depth_ee_pos,
    )


def distributed_barrier() -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def freeze_action_head_base(action_head: L1RegressionActionHead) -> None:
    for param in action_head.model.parameters():
        param.requires_grad = False
    if action_head.depth_context is not None:
        for param in action_head.depth_context.parameters():
            param.requires_grad = True
    if action_head.depth_action_residual is not None:
        for param in action_head.depth_action_residual.parameters():
            param.requires_grad = True
    if getattr(action_head, "depth_keypose_action_residual", None) is not None:
        for param in action_head.depth_keypose_action_residual.parameters():
            param.requires_grad = True
    if action_head.spatial_head is not None:
        for param in action_head.spatial_head.parameters():
            param.requires_grad = True
    if getattr(action_head, "object_query_spatial_heads", None) is not None:
        for param in action_head.object_query_spatial_heads.parameters():
            param.requires_grad = True
    if getattr(action_head, "depth_point_score_head", None) is not None:
        for param in action_head.depth_point_score_head.parameters():
            param.requires_grad = True
    if getattr(action_head, "depth_point_action_residual", None) is not None:
        for param in action_head.depth_point_action_residual.parameters():
            param.requires_grad = True
    if hasattr(action_head, "depth_fusion_gate"):
        action_head.depth_fusion_gate.requires_grad = True


def configure_depth_alpha(
    depth_encoder: LightweightDepthTokenEncoder,
    alpha_init: Optional[float],
    freeze_alpha: bool,
    min_alpha: Optional[float],
) -> None:
    if alpha_init is not None:
        with torch.no_grad():
            depth_encoder.alpha.fill_(float(alpha_init))
    if min_alpha is not None:
        with torch.no_grad():
            depth_encoder.alpha.clamp_(min=float(min_alpha))
    depth_encoder.alpha.requires_grad = not freeze_alpha


def clamp_depth_alpha(depth_encoder: nn.Module | None, min_alpha: Optional[float]) -> None:
    if depth_encoder is None or min_alpha is None:
        return
    with torch.no_grad():
        unwrap_module(depth_encoder).alpha.clamp_(min=float(min_alpha))


def summarize_trainable_parameters(module: nn.Module, name: str, max_names: int = 12) -> None:
    trainable = [(param_name, p.numel()) for param_name, p in module.named_parameters() if p.requires_grad]
    print(f"# trainable parameter tensors in {name}: {len(trainable)}")
    for param_name, num_params in trainable[:max_names]:
        print(f"  - {name}.{param_name}: {num_params}")
    if len(trainable) > max_names:
        print(f"  - ... {len(trainable) - max_names} more tensors")


def normalize_depth_integration_mode(
    depth_integration_mode: Optional[str], use_depth: bool, depth_fusion_mode: str
) -> str:
    if depth_integration_mode is None:
        if not use_depth:
            return "rgb_only"
        if depth_fusion_mode == "prefix":
            return "depth_prefix_append"
        if depth_fusion_mode == "action_head":
            return "depth_action_fusion"
        if depth_fusion_mode == "action_residual":
            return "depth_action_residual"
        if depth_fusion_mode == "action_summary_aux":
            return "depth_action_summary_aux"
        if depth_fusion_mode == "object_query":
            return "depth_object_query"
        raise ValueError(f"Unknown depth_fusion_mode: {depth_fusion_mode}")

    aliases = {
        "rgb": "rgb_only",
        "rgb_only": "rgb_only",
        "none": "rgb_only",
        "prefix": "depth_prefix_append",
        "depth_prefix_append": "depth_prefix_append",
        "action_head": "depth_action_fusion",
        "depth_action_fusion": "depth_action_fusion",
        "action_residual": "depth_action_residual",
        "residual": "depth_action_residual",
        "depth_action_residual": "depth_action_residual",
        "action_summary_aux": "depth_action_summary_aux",
        "summary_aux": "depth_action_summary_aux",
        "depth_action_summary_aux": "depth_action_summary_aux",
        "object_query": "depth_object_query",
        "depth_object_query": "depth_object_query",
    }
    try:
        return aliases[depth_integration_mode]
    except KeyError as exc:
        valid = "rgb_only|depth_prefix_append|depth_action_fusion|depth_action_residual|depth_action_summary_aux|depth_object_query"
        raise ValueError(f"Unknown depth_integration_mode: {depth_integration_mode}; expected {valid}") from exc


def apply_depth_integration_mode(cfg: DepthFinetuneConfig) -> str:
    mode = normalize_depth_integration_mode(cfg.depth_integration_mode, cfg.use_depth, cfg.depth_fusion_mode)
    cfg.depth_integration_mode = mode
    cfg.use_depth = mode != "rgb_only"
    if mode == "depth_action_fusion":
        cfg.depth_fusion_mode = "action_head"
    elif mode == "depth_action_residual":
        cfg.depth_fusion_mode = "action_residual"
    elif mode == "depth_action_summary_aux":
        cfg.depth_fusion_mode = "action_summary_aux"
    elif mode == "depth_object_query":
        cfg.depth_fusion_mode = "object_query"
    else:
        cfg.depth_fusion_mode = "prefix"
    return mode


def apply_v1_summary_config(cfg: DepthFinetuneConfig) -> None:
    """Keep the active experiment line on the v1 action-summary representation."""
    if cfg.depth_integration_mode != "depth_action_summary_aux":
        return
    if cfg.summary_repr != "base_xyz":
        raise ValueError(
            "DepthVLA action-summary has been restored to the v1 base_xyz configuration. "
            "Use --summary_repr base_xyz or omit --summary_repr."
        )
    cfg.summary_pool = "meanmax"


def get_run_id(cfg: DepthFinetuneConfig) -> str:
    if cfg.run_id_override is not None:
        return cfg.run_id_override
    integration_mode = normalize_depth_integration_mode(
        cfg.depth_integration_mode, cfg.use_depth, cfg.depth_fusion_mode
    )
    if integration_mode == "rgb_only":
        depth_tag = "rgb-only"
    elif cfg.depth_encoder_type == "dense_point":
        depth_tag = f"depth-densep{cfg.depth_num_points_per_view}"
    else:
        depth_tag = f"depth-g{cfg.depth_grid_size}"
    if integration_mode == "depth_action_fusion":
        depth_tag += "+fusion-action"
    if integration_mode == "depth_action_residual":
        depth_tag += "+action-residual"
    if integration_mode == "depth_action_summary_aux":
        depth_tag += "+action-summary-aux"
    if integration_mode == "depth_object_query":
        depth_tag += "+object-query"
        if cfg.summary_repr != "base_xyz":
            depth_tag += f"+repr-{cfg.summary_repr}"
    if integration_mode != "rgb_only" and cfg.geometry_norm != "none":
        depth_tag += f"+geom-{cfg.geometry_norm}+clip-{cfg.geometry_clip}"
    if integration_mode in ("depth_action_fusion", "depth_action_residual", "depth_action_summary_aux", "depth_object_query"):
        depth_tag += f"+gate-{cfg.depth_action_fusion_gate_init}"
        if cfg.depth_keypose_residual_weight > 0:
            depth_tag += f"+kpres-{cfg.depth_keypose_residual_weight}"
            if cfg.depth_keypose_residual_clip > 0:
                depth_tag += f"+kpclip-{cfg.depth_keypose_residual_clip}"
        if cfg.depth_point_action_weight > 0:
            depth_tag += f"+ptact-{cfg.depth_point_action_weight}"
            if cfg.depth_point_action_clip > 0:
                depth_tag += f"+ptclip-{cfg.depth_point_action_clip}"
        if cfg.depth_waypoint_action_weight > 0:
            depth_tag += f"+wpact-{cfg.depth_waypoint_action_weight}"
            if cfg.depth_waypoint_action_clip > 0:
                depth_tag += f"+wpclip-{cfg.depth_waypoint_action_clip}"
            if cfg.depth_waypoint_action_scale != 1.0:
                depth_tag += f"+wpscale-{cfg.depth_waypoint_action_scale}"
            if cfg.depth_waypoint_action_chunk_len != 1:
                depth_tag += f"+wpchunk-{cfg.depth_waypoint_action_chunk_len}"
        if cfg.depth_aux_spatial_loss_weight > 0 and cfg.aux_target != "none":
            depth_tag += f"+aux-{short_run_tag(cfg.aux_target)}-{cfg.depth_aux_spatial_loss_weight}"
    run_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}+{depth_tag}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return shorten_run_id(run_id)


def read_hdf5_string(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def standardize_actions(actions: np.ndarray, source: str = "libero") -> np.ndarray:
    actions = actions.astype(np.float32).copy()
    source = str(source or "libero").lower()
    if source == "rlbench":
        # RLBench converter already writes [dx,dy,dz,droll,dpitch,dyaw,gripper_open].
        # Do not apply LIBERO's gripper convention flip.
        return actions.astype(np.float32)
    gripper = 1.0 - np.clip(actions[:, -1:], 0.0, 1.0)
    return np.concatenate([actions[:, :6], gripper], axis=1).astype(np.float32)


def get_proprio(obs_group) -> np.ndarray:
    if "proprio" in obs_group:
        return obs_group["proprio"][()].astype(np.float32)
    ee_states = obs_group["ee_states"][()].astype(np.float32)
    gripper_states = obs_group["gripper_states"][()].astype(np.float32)
    return np.concatenate([ee_states, gripper_states], axis=1).astype(np.float32)


def maybe_rotate_policy_rgb(rgb: np.ndarray, source: str) -> np.ndarray:
    source = str(source or "libero").lower()
    if source == "rlbench":
        return rgb
    return rgb[::-1, ::-1]


def clean_sim_object_name(name: str) -> str:
    if isinstance(name, bytes):
        name = name.decode()
    name = str(name or "").strip()
    if not name:
        return ""
    parts = [part for part in name.split("_") if part]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)


def append_object_context_to_instruction(instruction: str, demo) -> str:
    manipulated = clean_sim_object_name(demo.attrs.get("manipulated_object_name", ""))
    target = clean_sim_object_name(demo.attrs.get("target_object_name", ""))
    context_parts = []
    if manipulated:
        context_parts.append(f"manipulated object: {manipulated}")
    if target:
        context_parts.append(f"target object: {target}")
    if not context_parts:
        return instruction
    return f"{instruction}. " + "; ".join(context_parts)


AUX_TARGET_CHOICES = {
    "none",
    "next_action_xyz",
    "relative_xyz",
    "contact_xyz",
    "visible_object_rel_xyz",
    "visible_object_point_xyz",
    "ee_to_object_xyz",
    "object_to_target_xyz",
    "gripper_to_contact_distance",
    "task_3d",
    "absolute_keypose",
    "rlbench_keypose_action",
    "point_keypose_xyz",
    "first_close_pose_xyz",
    "pre_first_close_pose_xyz",
    "visible_first_close_point_xyz",
    "visible_pre_first_close_point_xyz",
    "future_pose_xyz",
    "final_pose_xyz",
    "farthest_future_pose_xyz",
    "projected_keypose_uv",
    "projected_keypose_heatmap",
    "relative_z_bin",
    "distance_bin",
}


RUN_TAG_AUX_TARGET_ALIASES = {
    "visible_first_close_point_xyz": "vis-first-close-pt",
    "visible_pre_first_close_point_xyz": "vis-preclose-pt",
    "first_close_pose_xyz": "first-close",
    "pre_first_close_pose_xyz": "preclose",
    "farthest_future_pose_xyz": "far-future",
    "projected_keypose_heatmap": "proj-kp-hm",
    "projected_keypose_uv": "proj-kp-uv",
    "visible_object_point_xyz": "vis-obj-pt",
}


def short_run_tag(value: str) -> str:
    return RUN_TAG_AUX_TARGET_ALIASES.get(str(value), str(value))


def shorten_run_id(run_id: str, max_len: int = 220) -> str:
    """Keep run directory components below common 255-byte filesystem limits."""
    if len(run_id.encode("utf-8")) <= max_len:
        return run_id
    digest = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:8]
    prefix = run_id.encode("utf-8")[: max_len - len(digest) - 1].decode("utf-8", errors="ignore")
    return f"{prefix}-{digest}"


def parse_aux_bin_edges(value: str | Sequence[float]) -> np.ndarray:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        edges = np.asarray([float(p) for p in parts], dtype=np.float32)
    else:
        edges = np.asarray(list(value), dtype=np.float32)
    if edges.shape != (2,):
        raise ValueError(f"Expected exactly two aux bin edges for 3 classes, got {edges}")
    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"Aux bin edges must be strictly increasing, got {edges}")
    return edges


def safe_signed_focal(value: float, eps: float = 1e-6) -> float:
    value = float(value)
    if abs(value) >= eps:
        return value
    return -eps if value < 0 else eps


def compute_visible_geometry_relative_xyz(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    ee_pos: np.ndarray,
    stride: int = 8,
) -> np.ndarray:
    """Approximate gripper-to-nearest-visible-object vector from metric depth.

    The regenerated HDF5 files do not store symbolic object poses. This derives a
    geometry-only proxy from agentview depth by back-projecting a sparse point
    cloud to the LIBERO base frame and selecting the nearest visible point in a
    conservative tabletop workspace. It is intentionally independent of next
    action labels, making it harder to satisfy via action priors alone.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = depth[::stride, ::stride]
    height, width = depth.shape
    ys, xs = np.meshgrid(
        np.arange(0, depth_m.shape[0], stride, dtype=np.float32),
        np.arange(0, depth_m.shape[1], stride, dtype=np.float32),
        indexing="ij",
    )
    xs = xs[:height, :width]
    ys = ys[:height, :width]

    fx = safe_signed_focal(float(intrinsics[0, 0]))
    fy = safe_signed_focal(float(intrinsics[1, 1]))
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    valid = np.isfinite(depth) & (depth >= 0.01) & (depth <= 5.0)

    z_cam = depth
    x_cam = (xs - cx) * z_cam / fx
    y_cam = (ys - cy) * z_cam / fy
    xyz1_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)
    xyz_base = np.einsum("ij,hwj->hwi", extrinsics.astype(np.float32), xyz1_cam)[..., :3]

    workspace = (
        valid
        & (xyz_base[..., 2] > 0.75)
        & (xyz_base[..., 2] < 1.25)
        & (xyz_base[..., 0] > -0.3)
        & (xyz_base[..., 0] < 1.0)
        & (xyz_base[..., 1] > -0.8)
        & (xyz_base[..., 1] < 0.8)
    )
    points = xyz_base[workspace]
    if points.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    ee = np.asarray(ee_pos, dtype=np.float32).reshape(1, 3)
    deltas = points - ee
    idx = np.linalg.norm(deltas, axis=1).argmin()
    return deltas[idx].astype(np.float32)


def compute_visible_geometry_point_xyz(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    ee_pos: np.ndarray,
    stride: int = 8,
) -> np.ndarray:
    rel_xyz = compute_visible_geometry_relative_xyz(depth_m, intrinsics, extrinsics, ee_pos, stride=stride)
    return (np.asarray(ee_pos, dtype=np.float32).reshape(3) + rel_xyz).astype(np.float32)


def backproject_visible_workspace_points(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    stride: int = 2,
) -> np.ndarray:
    """Back-project a sparse visible point cloud in the robot base frame."""
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    full_height, full_width = depth.shape
    depth_ds = depth[::stride, ::stride]
    height, width = depth_ds.shape
    ys, xs = np.meshgrid(
        np.arange(0, full_height, stride, dtype=np.float32),
        np.arange(0, full_width, stride, dtype=np.float32),
        indexing="ij",
    )
    xs = xs[:height, :width]
    ys = ys[:height, :width]

    fx = safe_signed_focal(float(intrinsics[0, 0]))
    fy = safe_signed_focal(float(intrinsics[1, 1]))
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    valid = np.isfinite(depth_ds) & (depth_ds >= 0.01) & (depth_ds <= 5.0)

    z_cam = depth_ds
    x_cam = (xs - cx) * z_cam / fx
    y_cam = (ys - cy) * z_cam / fy
    xyz1_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)
    xyz_base = np.einsum("ij,hwj->hwi", extrinsics.astype(np.float32), xyz1_cam)[..., :3]

    workspace = (
        valid
        & (xyz_base[..., 2] > 0.65)
        & (xyz_base[..., 2] < 1.35)
        & (xyz_base[..., 0] > -0.4)
        & (xyz_base[..., 0] < 1.1)
        & (xyz_base[..., 1] > -0.9)
        & (xyz_base[..., 1] < 0.9)
    )
    return xyz_base[workspace].astype(np.float32)


def compute_visible_point_near_xyz(
    ep: Dict,
    t: int,
    target_xyz: np.ndarray,
    stride: int = 2,
) -> np.ndarray:
    """Return the current visible 3D point closest to a target base-frame XYZ."""
    target = np.asarray(target_xyz, dtype=np.float32).reshape(1, 3)
    point_sets = []
    for view in ("agentview", "eye_in_hand"):
        point_sets.append(
            backproject_visible_workspace_points(
                ep[f"{view}_depth_m"][t],
                ep[f"{view}_K"][t],
                ep[f"{view}_T_camera_to_base"][t],
                stride=stride,
            )
        )
    points = np.concatenate([pts for pts in point_sets if pts.size > 0], axis=0) if any(
        pts.size > 0 for pts in point_sets
    ) else np.empty((0, 3), dtype=np.float32)
    if points.shape[0] == 0:
        return target.reshape(3).astype(np.float32)
    idx = int(np.linalg.norm(points - target, axis=1).argmin())
    return points[idx].astype(np.float32)


def project_point_to_pixel_uv(point_xyz: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> tuple[float, float, bool]:
    """Project a base-frame point to pixel UV in a camera frame."""
    point_xyz = np.asarray(point_xyz, dtype=np.float32).reshape(3)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    try:
        t_base_to_cam = np.linalg.inv(extrinsics.astype(np.float64)).astype(np.float32)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, False
    point_cam = (t_base_to_cam @ np.asarray([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=np.float32))[:3]
    if not np.isfinite(point_cam).all() or point_cam[2] <= 0.01:
        return 0.0, 0.0, False
    u = float(intrinsics[0, 0] * point_cam[0] / point_cam[2] + intrinsics[0, 2])
    v = float(intrinsics[1, 1] * point_cam[1] / point_cam[2] + intrinsics[1, 2])
    return u, v, True


def project_point_to_normalized_uv(point_xyz: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Project a base-frame point to normalized camera UV in [-1, 1]."""
    height, width = int(image_hw[0]), int(image_hw[1])
    u, v, valid = project_point_to_pixel_uv(point_xyz, intrinsics, extrinsics)
    if not valid:
        return np.zeros(2, dtype=np.float32)
    u_norm = 2.0 * (u / max(width - 1, 1)) - 1.0
    v_norm = 2.0 * (v / max(height - 1, 1)) - 1.0
    return np.asarray([u_norm, v_norm], dtype=np.float32).clip(-2.0, 2.0)


def compute_projected_keypose_uv_label(ep: Dict, t: int) -> np.ndarray:
    if "rlbench_keypose_action" not in ep:
        raise KeyError("aux_target='projected_keypose_uv' requires top-level dataset rlbench_keypose_action")
    point_xyz = ep["rlbench_keypose_action"][t, :3].astype(np.float32)
    parts = []
    for view in ("agentview", "eye_in_hand"):
        depth = ep[f"{view}_depth_m"][t]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            image_hw = depth.shape[:2]
        else:
            image_hw = depth.shape[-2:]
        parts.append(
            project_point_to_normalized_uv(
                point_xyz,
                ep[f"{view}_K"][t],
                ep[f"{view}_T_camera_to_base"][t],
                image_hw=image_hw,
            )
        )
    return np.concatenate(parts, axis=0).astype(np.float32)


def gaussian_heatmap_label(size: int, u: float, v: float, sigma: float) -> np.ndarray:
    y = np.arange(size, dtype=np.float32)
    x = np.arange(size, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    heatmap = np.exp(-((xx - float(u)) ** 2 + (yy - float(v)) ** 2) / (2.0 * float(sigma) ** 2))
    max_value = float(heatmap.max())
    if max_value > 1e-8:
        heatmap = heatmap / max_value
    return heatmap.astype(np.float32)


def compute_projected_keypose_heatmap_label(ep: Dict, t: int, heatmap_size: int, sigma: float) -> np.ndarray:
    if "rlbench_keypose_action" not in ep:
        raise KeyError("aux_target='projected_keypose_heatmap' requires top-level dataset rlbench_keypose_action")
    point_xyz = ep["rlbench_keypose_action"][t, :3].astype(np.float32)
    heatmap_size = int(heatmap_size)
    heatmaps = []
    for view in ("agentview", "eye_in_hand"):
        depth = ep[f"{view}_depth_m"][t]
        image_hw = depth.shape[:2] if depth.ndim == 3 and depth.shape[-1] == 1 else depth.shape[-2:]
        height, width = int(image_hw[0]), int(image_hw[1])
        u, v, valid = project_point_to_pixel_uv(point_xyz, ep[f"{view}_K"][t], ep[f"{view}_T_camera_to_base"][t])
        u_map = u * heatmap_size / max(width, 1)
        v_map = v * heatmap_size / max(height, 1)
        valid = bool(valid and 0.0 <= u_map < heatmap_size and 0.0 <= v_map < heatmap_size)
        if valid:
            heatmaps.append(gaussian_heatmap_label(heatmap_size, u_map, v_map, sigma))
        else:
            heatmaps.append(np.zeros((heatmap_size, heatmap_size), dtype=np.float32))
    return np.stack(heatmaps, axis=0).astype(np.float32)


def compute_rlbench_xyz_aux_label(ep: Dict, t: int, aux_target: str, future_horizon: int = 10) -> np.ndarray:
    """Return absolute XYZ labels for RLBench spatial-action supervision."""
    aux_target = str(aux_target or "")
    if aux_target == "point_keypose_xyz":
        if "rlbench_keypose_action" not in ep:
            raise KeyError("aux_target='point_keypose_xyz' requires top-level dataset rlbench_keypose_action")
        return ep["rlbench_keypose_action"][t, :3].astype(np.float32)

    if "rlbench_abs_gripper_pose" not in ep:
        raise KeyError(
            f"aux_target={aux_target!r} requires obs/rlbench_abs_gripper_pose from converted RLBench HDF5s"
        )
    abs_xyz = ep["rlbench_abs_gripper_pose"][:, :3].astype(np.float32)
    if aux_target in (
        "first_close_pose_xyz",
        "pre_first_close_pose_xyz",
        "visible_first_close_point_xyz",
        "visible_pre_first_close_point_xyz",
    ):
        if "proprio" not in ep:
            raise KeyError(f"aux_target={aux_target!r} requires obs/proprio gripper-open state")
        gripper_open = ep["proprio"][:, -1].astype(np.float32)
        close_indices = np.flatnonzero(gripper_open < 0.5)
        close_index = int(close_indices[0]) if close_indices.size else abs_xyz.shape[0] - 1
        if aux_target in ("pre_first_close_pose_xyz", "visible_pre_first_close_point_xyz"):
            close_index = max(0, close_index - max(1, int(future_horizon)))
        target_xyz = abs_xyz[close_index].astype(np.float32)
        if aux_target in ("visible_first_close_point_xyz", "visible_pre_first_close_point_xyz"):
            required = (
                "agentview_depth_m",
                "eye_in_hand_depth_m",
                "agentview_K",
                "eye_in_hand_K",
                "agentview_T_camera_to_base",
                "eye_in_hand_T_camera_to_base",
            )
            missing = [key for key in required if key not in ep]
            if missing:
                raise KeyError(f"aux_target={aux_target!r} requires current RGB-D geometry fields; missing {missing}")
            return compute_visible_point_near_xyz(ep, t, target_xyz).astype(np.float32)
        return target_xyz
    if aux_target == "future_pose_xyz":
        future_index = min(int(t) + max(1, int(future_horizon)), abs_xyz.shape[0] - 1)
        return abs_xyz[future_index].astype(np.float32)
    if aux_target == "final_pose_xyz":
        return abs_xyz[-1].astype(np.float32)
    if aux_target == "farthest_future_pose_xyz":
        ee_pos = ep["ee_pos"][t, :3].astype(np.float32)
        future_window = abs_xyz[int(t) :]
        farthest_index = int(np.argmax(np.linalg.norm(future_window - ee_pos.reshape(1, 3), axis=1)))
        return future_window[farthest_index].astype(np.float32)
    raise ValueError(f"Unknown RLBench xyz aux_target: {aux_target}")


def compute_aux_label_from_geometry(ep: Dict, t: int, aux_target: str, distance_edges: np.ndarray, z_edges: np.ndarray) -> np.ndarray:
    if aux_target == "none":
        return np.zeros((), dtype=np.int64)
    if aux_target == "next_action_xyz":
        raise ValueError("next_action_xyz aux target is computed after action normalization, not from geometry")
    if aux_target == "object_to_target_xyz":
        if "object_to_target_xyz" in ep:
            return ep["object_to_target_xyz"][t].astype(np.float32)
        raise KeyError(
            "aux_target='object_to_target_xyz' requires obs/object_to_target_xyz from regenerated RGB-D HDF5s. "
            "Regenerate LIBERO-Plus RGB-D data with symbolic object/target pose saving enabled."
        )
    if aux_target == "task_3d":
        required_fields = ("ee_to_object_xyz", "object_to_target_xyz", "gripper_to_contact_distance")
        missing = [field for field in required_fields if field not in ep]
        if missing:
            raise KeyError(
                f"aux_target='task_3d' requires regenerated RGB-D HDF5 obs fields {required_fields}; "
                f"missing {missing}. Regenerate LIBERO-Plus RGB-D data with symbolic object/target pose saving enabled."
            )
        return np.concatenate(
            [
                ep["ee_to_object_xyz"][t].astype(np.float32),
                ep["object_to_target_xyz"][t].astype(np.float32),
                ep["gripper_to_contact_distance"][t].astype(np.float32).reshape(1),
            ],
            axis=0,
        ).astype(np.float32)
    if aux_target == "ee_to_object_xyz" and "ee_to_object_xyz" in ep:
        return ep["ee_to_object_xyz"][t].astype(np.float32)
    if aux_target == "gripper_to_contact_distance" and "gripper_to_contact_distance" in ep:
        return ep["gripper_to_contact_distance"][t].astype(np.float32)

    rel_xyz = compute_visible_geometry_relative_xyz(
        ep["agentview_depth_m"][t],
        ep["agentview_K"][t],
        ep["agentview_T_camera_to_base"][t],
        ep["ee_pos"][t],
    )
    if aux_target == "visible_object_point_xyz":
        return compute_visible_geometry_point_xyz(
            ep["agentview_depth_m"][t],
            ep["agentview_K"][t],
            ep["agentview_T_camera_to_base"][t],
            ep["ee_pos"][t],
        )
    if aux_target in ("relative_xyz", "contact_xyz", "visible_object_rel_xyz", "ee_to_object_xyz"):
        return rel_xyz.astype(np.float32)
    if aux_target == "gripper_to_contact_distance":
        return np.asarray([np.linalg.norm(rel_xyz)], dtype=np.float32)
    if aux_target == "distance_bin":
        distance = float(np.linalg.norm(rel_xyz))
        return np.asarray(np.digitize(distance, distance_edges), dtype=np.int64)
    if aux_target == "relative_z_bin":
        return np.asarray(np.digitize(float(rel_xyz[2]), z_edges), dtype=np.int64)
    raise ValueError(f"Unknown aux_target: {aux_target}")


def bounds_q99_stats(values: np.ndarray, mask: Optional[List[bool]] = None) -> Dict:
    stats = {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }
    if mask is not None:
        stats["mask"] = mask
    return stats


def normalize_bounds_q99(values: np.ndarray, stats: Dict) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.where(mask, 2 * (values - low) / (high - low + 1e-8) - 1, values)
    normalized = np.where(mask, np.clip(normalized, -1.0, 1.0), normalized)
    normalized = np.where(mask & (low == high), 0.0, normalized)
    return normalized.astype(np.float32)


def summarize_array(values: np.ndarray) -> Dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "p1": np.percentile(values, 1, axis=0).tolist(),
        "p99": np.percentile(values, 99, axis=0).tolist(),
    }


def print_geometry_stats(title: str, stats: Dict) -> None:
    print(f"DepthVLA geometry normalization stats: {title}")
    for idx, name in enumerate(GEOMETRY_CONTINUOUS_FEATURE_NAMES):
        print(
            f"  {name}: mean={stats['mean'][idx]:.6f}, std={stats['std'][idx]:.6f}, "
            f"min={stats['min'][idx]:.6f}, max={stats['max'][idx]:.6f}, "
            f"p1={stats['p1'][idx]:.6f}, p99={stats['p99'][idx]:.6f}"
        )


def normalize_geometry_continuous(values: np.ndarray, stats: Dict, clip: float) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    normalized = (values.astype(np.float32) - mean) / (std + 1e-6)
    if clip is not None and clip > 0:
        normalized = np.clip(normalized, -float(clip), float(clip))
    return normalized


def compute_geometry_norm_stats(dataset, depth_encoder, geometry_clip: float, chunk_size: int = 16) -> Dict:
    module = depth_encoder.module if hasattr(depth_encoder, "module") else depth_encoder
    device = next(module.parameters()).device
    chunks = []
    before_examples, after_examples = None, None
    seen = []
    for file_path, demo_key, _, _ in dataset.samples:
        key = (file_path, demo_key)
        if key not in seen:
            seen.append(key)

    was_training = module.training
    module.eval()
    with torch.inference_mode():
        for file_path, demo_key in seen:
            ep = dataset._load_episode(file_path, demo_key)
            num_steps = ep["agentview_depth_m"].shape[0]
            for start in range(0, num_steps, chunk_size):
                end = min(start + chunk_size, num_steps)
                depth_values = np.stack(
                    [ep["agentview_depth_m"][start:end], ep["eye_in_hand_depth_m"][start:end]], axis=1
                ).astype(np.float32)
                depth_intrinsics = np.stack(
                    [ep["agentview_K"][start:end], ep["eye_in_hand_K"][start:end]], axis=1
                ).astype(np.float32)
                depth_extrinsics = np.stack(
                    [ep["agentview_T_camera_to_base"][start:end], ep["eye_in_hand_T_camera_to_base"][start:end]], axis=1
                ).astype(np.float32)
                depth_valid_mask = np.isfinite(depth_values)

                features = module.compute_geometry_features(
                    torch.from_numpy(depth_values).to(device),
                    torch.from_numpy(depth_intrinsics).to(device),
                    torch.from_numpy(depth_extrinsics).to(device),
                    torch.from_numpy(depth_valid_mask).to(device),
                )[..., :4]
                arr = features.detach().float().cpu().reshape(-1, 4).numpy()
                chunks.append(arr)
                if before_examples is None:
                    before_examples = arr[:8].copy()

    if was_training:
        module.train()

    values = np.concatenate(chunks, axis=0)
    stats = summarize_array(values)
    normalized = normalize_geometry_continuous(values, stats, geometry_clip)
    normalized_stats = summarize_array(normalized)
    if before_examples is not None:
        after_examples = normalize_geometry_continuous(before_examples, stats, geometry_clip)

    result = {
        "feature_names": list(GEOMETRY_CONTINUOUS_FEATURE_NAMES),
        "normalization": "dataset_std",
        "clip": geometry_clip,
        "num_values": int(values.shape[0]),
        "mean": stats["mean"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
        "p1": stats["p1"],
        "p99": stats["p99"],
        "normalized_summary": normalized_stats,
    }
    print_geometry_stats("raw", stats)
    print_geometry_stats("after dataset_std", normalized_stats)
    if before_examples is not None:
        print("DepthVLA geometry examples before normalization [X,Y,Z,z_cam]:")
        print(np.array2string(before_examples, precision=4, suppress_small=False))
        print("DepthVLA geometry examples after normalization [X,Y,Z,z_cam]:")
        print(np.array2string(after_examples, precision=4, suppress_small=False))
    return result


def save_geometry_norm_stats(stats: Dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved DepthVLA geometry normalization stats to: {path}")


class LiberoRGBDHDF5Dataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        dataset_name: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer,
        image_transform,
        prompt_builder_fn,
        use_depth: bool = True,
        use_proprio: bool = True,
        aux_target: str = "none",
        aux_heatmap_size: int = 16,
        aux_heatmap_sigma: float = 1.5,
        aux_future_horizon: int = 10,
        aux_distance_bin_edges: str | Sequence[float] = "0.036,0.065",
        aux_z_bin_edges: str | Sequence[float] = "-0.04,0.04",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.use_depth = use_depth
        self.use_proprio = use_proprio
        self.aux_target = str(aux_target or "none")
        if self.aux_target not in AUX_TARGET_CHOICES:
            raise ValueError(f"Unknown aux_target {self.aux_target}; choose from {sorted(AUX_TARGET_CHOICES)}")
        self.aux_heatmap_size = int(aux_heatmap_size)
        self.aux_heatmap_sigma = float(aux_heatmap_sigma)
        self.aux_future_horizon = int(aux_future_horizon)
        self.aux_distance_bin_edges = parse_aux_bin_edges(aux_distance_bin_edges)
        self.aux_z_bin_edges = parse_aux_bin_edges(aux_z_bin_edges)
        self.samples = []
        self.episode_cache = {}

        self._index_files()
        self.dataset_statistics = self._compute_dataset_statistics()

    def _index_files(self) -> None:
        hdf5_files = sorted(list(self.data_dir.glob("*.hdf5")) + list(self.data_dir.glob("*.h5")))
        if len(hdf5_files) == 0:
            raise FileNotFoundError(f"No HDF5 files found in {self.data_dir}")
        for file_path in hdf5_files:
            with h5py.File(file_path, "r") as f:
                for demo_key in sorted(f["data"].keys()):
                    demo = f["data"][demo_key]
                    length = demo["actions"].shape[0]
                    instruction = self._read_instruction(demo, file_path)
                    for t in range(length):
                        self.samples.append((str(file_path), demo_key, t, instruction))

    def _read_instruction(self, demo, file_path: Path) -> str:
        if "language_instruction" in demo.attrs:
            value = demo.attrs["language_instruction"]
            instruction = value.decode() if isinstance(value, bytes) else str(value)
            return append_object_context_to_instruction(instruction, demo)
        if "language_instruction" in demo:
            value = demo["language_instruction"][()]
            instruction = value.decode() if isinstance(value, bytes) else str(value)
            return append_object_context_to_instruction(instruction, demo)
        return file_path.stem.replace("_demo", "").replace("_", " ")

    def _load_episode(self, file_path: str, demo_key: str) -> Dict:
        cache_key = (file_path, demo_key)
        if cache_key in self.episode_cache:
            return self.episode_cache[cache_key]
        with h5py.File(file_path, "r") as f:
            source = read_hdf5_string(f.attrs.get("source", "libero"), default="libero").lower()
            demo = f["data"][demo_key]
            obs = demo["obs"]
            ep = {
                "source": source,
                "actions": standardize_actions(demo["actions"][()], source=source),
                "proprio": get_proprio(obs),
                "agentview_rgb": obs["agentview_rgb"][()],
                "eye_in_hand_rgb": obs["eye_in_hand_rgb"][()],
            }
            for key in ("rlbench_keypose_action", "rlbench_delta_action"):
                if key in demo:
                    ep[key] = demo[key][()].astype(np.float32)
            if "rlbench_abs_gripper_pose" in obs:
                ep["rlbench_abs_gripper_pose"] = obs["rlbench_abs_gripper_pose"][()].astype(np.float32)
            if "rlbench_next_abs_gripper_pose" in obs:
                ep["rlbench_next_abs_gripper_pose"] = obs["rlbench_next_abs_gripper_pose"][()].astype(np.float32)
            if "ee_pos" in obs:
                ep["ee_pos"] = obs["ee_pos"][()].astype(np.float32)
            else:
                ep["ee_pos"] = obs["ee_states"][()].astype(np.float32)[:, :3]
            for key in (
                "manipulated_object_pos",
                "target_pos",
                "ee_to_object_xyz",
                "object_to_target_xyz",
                "gripper_to_contact_distance",
            ):
                if key in obs:
                    ep[key] = obs[key][()].astype(np.float32)
            if self.use_depth:
                for key in (
                    "agentview_depth_m",
                    "eye_in_hand_depth_m",
                    "agentview_K",
                    "eye_in_hand_K",
                    "agentview_T_camera_to_base",
                    "eye_in_hand_T_camera_to_base",
                ):
                    if key not in obs:
                        raise KeyError(f"Missing depth field obs/{key} in {file_path}:{demo_key}")
                    ep[key] = obs[key][()]
        if len(self.episode_cache) > 16:
            self.episode_cache.clear()
        self.episode_cache[cache_key] = ep
        return ep

    def _compute_dataset_statistics(self) -> Dict:
        actions, proprios = [], []
        seen = set()
        for file_path, demo_key, _, _ in self.samples:
            if (file_path, demo_key) in seen:
                continue
            seen.add((file_path, demo_key))
            ep = self._load_episode(file_path, demo_key)
            actions.append(ep["actions"])
            proprios.append(ep["proprio"])

        all_actions = np.concatenate(actions, axis=0)
        all_proprios = np.concatenate(proprios, axis=0)
        if all_actions.shape[-1] != ACTION_DIM:
            raise ValueError(f"Dataset action dim {all_actions.shape[-1]} does not match ACTION_DIM={ACTION_DIM}")
        if all_proprios.shape[-1] != PROPRIO_DIM:
            raise ValueError(f"Dataset proprio dim {all_proprios.shape[-1]} does not match PROPRIO_DIM={PROPRIO_DIM}")
        return {
            self.dataset_name: {
                "action": bounds_q99_stats(all_actions, mask=[True] * 6 + [False]),
                "proprio": bounds_q99_stats(all_proprios, mask=[True] * PROPRIO_DIM),
                "num_transitions": int(all_actions.shape[0]),
                "num_trajectories": len(seen),
            }
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        file_path, demo_key, t, instruction = self.samples[idx]
        ep = self._load_episode(file_path, demo_key)
        stats = self.dataset_statistics[self.dataset_name]

        actions = ep["actions"]
        action_chunk = []
        for offset in range(NUM_ACTIONS_CHUNK):
            action_chunk.append(actions[min(t + offset, len(actions) - 1)])
        action_chunk = normalize_bounds_q99(np.stack(action_chunk, axis=0), stats["action"])

        prompt_builder = self.prompt_builder_fn("openvla")
        current_action_string = self.action_tokenizer(action_chunk[0])
        future_actions_string = "".join(self.action_tokenizer(action_chunk[1:]))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction.lower()}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX

        rgb = Image.fromarray(maybe_rotate_policy_rgb(ep["agentview_rgb"][t], ep.get("source", "libero"))).convert("RGB")
        wrist_rgb = Image.fromarray(
            maybe_rotate_policy_rgb(ep["eye_in_hand_rgb"][t], ep.get("source", "libero"))
        ).convert("RGB")
        pixel_values = self.image_transform(rgb)
        pixel_values_wrist = self.image_transform(wrist_rgb)

        out = {
            "pixel_values": pixel_values,
            "pixel_values_wrist": pixel_values_wrist,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": self.dataset_name,
            "actions": action_chunk,
        }
        if self.use_proprio:
            out["proprio"] = normalize_bounds_q99(ep["proprio"][t], stats["proprio"])
        if self.use_depth:
            depth_values = np.stack([ep["agentview_depth_m"][t], ep["eye_in_hand_depth_m"][t]], axis=0)
            depth_intrinsics = np.stack([ep["agentview_K"][t], ep["eye_in_hand_K"][t]], axis=0)
            depth_extrinsics = np.stack(
                [ep["agentview_T_camera_to_base"][t], ep["eye_in_hand_T_camera_to_base"][t]], axis=0
            )
            out["depth_values"] = depth_values.astype(np.float32)
            out["depth_intrinsics"] = depth_intrinsics.astype(np.float32)
            out["depth_extrinsics"] = depth_extrinsics.astype(np.float32)
            out["depth_valid_mask"] = np.isfinite(depth_values).astype(np.bool_)
            out["depth_ee_pos"] = ep["ee_pos"][t].astype(np.float32)
        if self.aux_target != "none":
            if self.aux_target == "next_action_xyz":
                out["aux_label"] = action_chunk[0, :3].astype(np.float32)
            elif self.aux_target in ("absolute_keypose", "rlbench_keypose_action"):
                if "rlbench_keypose_action" not in ep:
                    raise KeyError(
                        f"aux_target={self.aux_target!r} requires top-level dataset rlbench_keypose_action"
                    )
                out["aux_label"] = ep["rlbench_keypose_action"][t].astype(np.float32)
            elif self.aux_target in (
                "point_keypose_xyz",
                "first_close_pose_xyz",
                "pre_first_close_pose_xyz",
                "visible_first_close_point_xyz",
                "visible_pre_first_close_point_xyz",
                "future_pose_xyz",
                "final_pose_xyz",
                "farthest_future_pose_xyz",
            ):
                out["aux_label"] = compute_rlbench_xyz_aux_label(
                    ep, t, self.aux_target, future_horizon=self.aux_future_horizon
                )
            elif self.aux_target == "projected_keypose_uv":
                out["aux_label"] = compute_projected_keypose_uv_label(ep, t)
            elif self.aux_target == "projected_keypose_heatmap":
                out["aux_label"] = compute_projected_keypose_heatmap_label(
                    ep, t, heatmap_size=self.aux_heatmap_size, sigma=self.aux_heatmap_sigma
                )
            else:
                out["aux_label"] = compute_aux_label_from_geometry(
                    ep, t, self.aux_target, self.aux_distance_bin_edges, self.aux_z_bin_edges
                )
        return out


class DepthPaddedCollatorForActionPrediction:
    def __init__(self, model_max_length: int, pad_token_id: int, padding_side: str = "right") -> None:
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)

        pixel_values = torch.cat(
            (
                torch.stack([instance["pixel_values"] for instance in instances]),
                torch.stack([instance["pixel_values_wrist"] for instance in instances]),
            ),
            dim=1,
        )
        output = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": torch.stack([torch.from_numpy(np.copy(instance["actions"])) for instance in instances]),
            "dataset_names": [instance["dataset_name"] for instance in instances],
        }
        if "proprio" in instances[0]:
            output["proprio"] = torch.tensor(np.stack([instance["proprio"] for instance in instances]), dtype=torch.float32)
        else:
            output["proprio"] = None
        if "depth_values" in instances[0]:
            for key in ("depth_values", "depth_intrinsics", "depth_extrinsics", "depth_valid_mask", "depth_ee_pos"):
                output[key] = torch.tensor(np.stack([instance[key] for instance in instances]))
        if "aux_label" in instances[0]:
            labels = [instance["aux_label"] for instance in instances]
            first = np.asarray(labels[0])
            if first.shape == ():
                output["aux_label"] = torch.tensor(np.asarray(labels), dtype=torch.long)
            else:
                output["aux_label"] = torch.tensor(np.stack(labels), dtype=torch.float32)
        return output


def compute_smoothened_metrics(metrics_deques) -> dict:
    return {name: sum(deque) / len(deque) for name, deque in metrics_deques.items() if deque}


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    log_dict = {}
    for name, value in metrics.items():
        key = "Loss" if name == "loss_value" else name.replace("_", " ").title()
        log_dict[f"{prefix}/{key}"] = value
    wandb_entity.log(log_dict, step=step)


def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    depth_encoder,
    batch,
    action_tokenizer,
    device_id,
    use_proprio,
    use_depth,
    use_film,
    num_patches,
    depth_fusion_mode="prefix",
    depth_aux_spatial_loss_weight=0.0,
    depth_integration_mode="depth_prefix_append",
    aux_target="none",
    depth_dropout=0.0,
    use_contrastive=False,
    contrastive_weight=0.0,
    contrastive_margin=0.05,
    null_to_base_weight=0.0,
    corrupt_to_base_weight=0.0,
    corrupt_depth_mode="shuffle_depth",
    log_diagnostics=False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    depth_context = None
    depth_context_full = None
    depth_point_features = None
    depth_point_features_full = None
    depth_keep_rate = None
    depth_kwargs = {}
    if use_depth:
        depth_values = batch["depth_values"].to(device_id)
        depth_intrinsics = batch["depth_intrinsics"].to(device_id)
        depth_extrinsics = batch["depth_extrinsics"].to(device_id)
        depth_valid_mask = batch["depth_valid_mask"].to(device_id)
        depth_ee_pos = batch.get("depth_ee_pos")
        if depth_ee_pos is not None:
            depth_ee_pos = depth_ee_pos.to(device_id)
        depth_kwargs = {
            "depth_values": depth_values,
            "depth_intrinsics": depth_intrinsics,
            "depth_extrinsics": depth_extrinsics,
            "depth_valid_mask": depth_valid_mask,
            "depth_ee_pos": depth_ee_pos,
            "depth_encoder": depth_encoder,
        }
        if depth_fusion_mode in ("action_head", "action_residual", "action_summary_aux", "object_query"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if depth_fusion_mode == "action_summary_aux":
                    depth_context = unwrap_module(depth_encoder).forward_summary(
                        depth_values=depth_values,
                        depth_intrinsics=depth_intrinsics,
                        depth_extrinsics=depth_extrinsics,
                        depth_valid_mask=depth_valid_mask,
                        ee_pos=depth_ee_pos,
                    )
                else:
                    depth_context = encode_depth_context(
                        depth_encoder=depth_encoder,
                        depth_values=depth_values,
                        depth_intrinsics=depth_intrinsics,
                        depth_extrinsics=depth_extrinsics,
                        depth_valid_mask=depth_valid_mask,
                        depth_ee_pos=depth_ee_pos,
                    )
            depth_context_full = depth_context
            depth_point_features_full = compute_depth_point_features(
                depth_encoder=depth_encoder,
                depth_values=depth_values,
                depth_intrinsics=depth_intrinsics,
                depth_extrinsics=depth_extrinsics,
                depth_valid_mask=depth_valid_mask,
                depth_ee_pos=depth_ee_pos,
            )
            depth_point_features = depth_point_features_full
            if vla.training and depth_dropout > 0:
                keep = torch.rand(depth_context.shape[0], device=depth_context.device) > float(depth_dropout)
                mask_shape = (depth_context.shape[0],) + (1,) * (depth_context.ndim - 1)
                depth_context = depth_context * keep.to(depth_context.dtype).view(mask_shape)
                if depth_point_features is not None:
                    point_mask_shape = (depth_point_features.shape[0],) + (1,) * (depth_point_features.ndim - 1)
                    depth_point_features = depth_point_features * keep.to(depth_point_features.dtype).view(point_mask_shape)
                depth_keep_rate = keep.float().mean()
            depth_kwargs = {}
        elif depth_fusion_mode != "prefix":
            raise ValueError(f"Unknown depth_fusion_mode: {depth_fusion_mode}")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"].to(device_id),
            output_hidden_states=True,
            proprio=batch["proprio"].to(device_id) if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            use_film=use_film,
            depth_fusion_mode=depth_fusion_mode,
            **depth_kwargs,
        )

    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)

    last_hidden_states = output.hidden_states[-1]
    text_hidden_states = last_hidden_states[:, num_patches:-1]
    batch_size = batch["input_ids"].shape[0]
    prompt_mask = batch["attention_mask"][:, 1:].to(device_id).bool() & ~(current_action_mask | next_actions_mask)
    prompt_mask_f = prompt_mask.to(text_hidden_states.dtype).unsqueeze(-1)
    prompt_context = (text_hidden_states * prompt_mask_f).sum(dim=1) / prompt_mask_f.sum(dim=1).clamp_min(1.0)
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )
    if log_diagnostics:
        print(f"DepthVLA integration mode: {depth_integration_mode}")
        print(f"DepthVLA depth_fusion_mode: {depth_fusion_mode}")
        if depth_encoder is not None:
            depth_encoder_module = unwrap_module(depth_encoder)
            print(f"DepthVLA summary_repr: {getattr(depth_encoder_module, 'summary_repr', 'base_xyz')}")
            print(f"DepthVLA summary_pool: {getattr(depth_encoder_module, 'summary_pool', 'meanmax')}")
            print(f"DepthVLA summary feature dim: {getattr(depth_encoder_module, 'summary_feature_dim', 8)}")
            print(f"DepthVLA depth alpha: {depth_encoder_module.alpha.detach().float().item():.6g}")
        print(f"DepthVLA appends depth tokens to prefix: {use_depth and depth_fusion_mode == 'prefix'}")
        print(f"DepthVLA action hidden shape before fusion: {tuple(actions_hidden_states.shape)}")
        if depth_context is not None:
            action_head_module = unwrap_module(action_head)
            pooled_depth_context = action_head_module.pool_depth_context(depth_context)
            with torch.no_grad():
                if getattr(action_head_module, "depth_fusion_type", None) == "action_residual":
                    residual = action_head_module.predict_action_residual(actions_hidden_states, depth_context)
                    fused_actions_hidden_states = actions_hidden_states
                else:
                    residual = None
                    fused_actions_hidden_states = action_head_module.condition_action_hidden_states(
                        actions_hidden_states, depth_context=depth_context, query_context=prompt_context
                    )
                keypose_residual = action_head_module.predict_keypose_action_residual(
                    actions_hidden_states, depth_context=depth_context, query_context=prompt_context
                )
            print(f"DepthVLA depth token/context shape: {tuple(depth_context.shape)}")
            if getattr(action_head_module, "depth_fusion_type", None) == "action_summary_aux":
                print(f"DepthVLA depth summary embedding shape: {tuple(pooled_depth_context.shape)}")
            else:
                print(f"DepthVLA pooled geometry embedding shape: {tuple(pooled_depth_context.shape)}")
            print(f"DepthVLA action hidden shape after fusion: {tuple(fused_actions_hidden_states.shape)}")
            if residual is not None:
                print(f"DepthVLA action residual abs mean: {residual.float().abs().mean().item():.6g}")
            if keypose_residual is not None:
                print(f"DepthVLA keypose action residual abs mean: {keypose_residual.float().abs().mean().item():.6g}")
            point_action_residual = action_head_module.predict_point_action_residual(
                actions_hidden_states,
                depth_context=depth_context,
                depth_point_features=depth_point_features,
                query_context=prompt_context,
            )
            if point_action_residual is not None:
                print(f"DepthVLA point action residual abs mean: {point_action_residual.float().abs().mean().item():.6g}")
            waypoint_xyz = action_head_module.predict_waypoint_xyz_action(
                actions_hidden_states,
                depth_context=depth_context,
                depth_point_features=depth_point_features,
                query_context=prompt_context,
            )
            if waypoint_xyz is not None:
                print(f"DepthVLA waypoint xyz action abs mean: {waypoint_xyz.float().abs().mean().item():.6g}")
            if depth_encoder is not None and getattr(unwrap_module(depth_encoder), "summary_repr", "base_xyz") == "ee_relative_set_v2":
                depth_encoder_module = unwrap_module(depth_encoder)
                with torch.no_grad():
                    old_mode = getattr(depth_encoder_module, "ablation_mode", "none")
                    summaries = {}
                    for mode in ("none", "null", "shuffle_tokens"):
                        depth_encoder_module.ablation_mode = mode
                        summaries[mode] = depth_encoder_module.forward_summary(
                            depth_values=depth_values,
                            depth_intrinsics=depth_intrinsics,
                            depth_extrinsics=depth_extrinsics,
                            depth_valid_mask=depth_valid_mask,
                            ee_pos=depth_ee_pos,
                        ).float()
                    depth_encoder_module.ablation_mode = old_mode
                print(f"DepthVLA summary normal norm: {summaries['none'].norm(dim=-1).mean().item():.6g}")
                print(f"DepthVLA summary null norm: {summaries['null'].norm(dim=-1).mean().item():.6g}")
                print(f"DepthVLA summary shuffle norm: {summaries['shuffle_tokens'].norm(dim=-1).mean().item():.6g}")
                print(f"DepthVLA summary normal-null L2: {(summaries['none'] - summaries['null']).norm(dim=-1).mean().item():.6g}")
                print(f"DepthVLA summary normal-shuffle L2: {(summaries['none'] - summaries['shuffle_tokens']).norm(dim=-1).mean().item():.6g}")
            print(f"DepthVLA action fusion gate: {action_head_module.depth_fusion_gate.detach().float().item():.6g}")
        elif use_depth:
            print("DepthVLA action-side depth context: none; depth is handled by prefix append mode")
        else:
            print("DepthVLA action-side depth context: none; RGB-only mode")

    action_head_module = unwrap_module(action_head)
    predicted_actions = action_head_module.predict_action(
        actions_hidden_states,
        depth_context=depth_context,
        query_context=prompt_context,
        depth_point_features=depth_point_features,
    )
    keypose_action_residual = None
    point_action_residual = None
    if depth_context is not None and getattr(action_head_module, "depth_keypose_action_residual", None) is not None:
        with torch.no_grad():
            keypose_action_residual = action_head_module.predict_keypose_action_residual(
                actions_hidden_states, depth_context=depth_context, query_context=prompt_context
            )
    if depth_context is not None and getattr(action_head_module, "depth_point_action_residual", None) is not None:
        with torch.no_grad():
            point_action_residual = action_head_module.predict_point_action_residual(
                actions_hidden_states,
                depth_context=depth_context,
                depth_point_features=depth_point_features,
                query_context=prompt_context,
            )
    action_loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)
    loss = action_loss

    contrastive_loss = None
    contrastive_loss_null = None
    contrastive_loss_corrupt = None
    action_loss_normal_full = None
    action_loss_null = None
    null_to_base_loss = None
    corrupt_to_base_loss = None
    action_loss_corrupt = None
    if (
        vla.training
        and depth_context_full is not None
        and depth_fusion_mode in ("action_head", "action_residual", "action_summary_aux", "object_query")
    ):
        predicted_actions_normal_full = None
        predicted_actions_null = None
        predicted_actions_base = None
        predicted_actions_corrupt = None

        def get_normal_full_actions():
            nonlocal predicted_actions_normal_full
            if predicted_actions_normal_full is None:
                predicted_actions_normal_full = action_head_module.predict_action(
                    actions_hidden_states,
                    depth_context=depth_context_full,
                    query_context=prompt_context,
                    depth_point_features=depth_point_features_full,
                )
            return predicted_actions_normal_full

        def get_null_actions():
            nonlocal predicted_actions_null
            if predicted_actions_null is None:
                predicted_actions_null = action_head_module.predict_action(
                    actions_hidden_states,
                    depth_context=torch.zeros_like(depth_context_full),
                    query_context=prompt_context,
                    depth_point_features=torch.zeros_like(depth_point_features_full)
                    if depth_point_features_full is not None
                    else None,
                )
            return predicted_actions_null

        def get_base_actions():
            nonlocal predicted_actions_base
            if predicted_actions_base is None:
                predicted_actions_base = action_head_module.predict_action(
                    actions_hidden_states,
                    depth_context=None,
                    query_context=prompt_context,
                    depth_point_features=None,
                )
            return predicted_actions_base

        def get_corrupt_actions():
            nonlocal predicted_actions_corrupt
            if predicted_actions_corrupt is None:
                if depth_encoder is None:
                    raise ValueError("corrupt depth actions require depth_encoder")
                depth_encoder_module = unwrap_module(depth_encoder)
                old_mode = getattr(depth_encoder_module, "ablation_mode", "none")
                try:
                    depth_encoder_module.ablation_mode = str(corrupt_depth_mode or "shuffle_depth")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if depth_fusion_mode == "action_summary_aux":
                            corrupt_depth_context = depth_encoder_module.forward_summary(
                                depth_values=depth_values,
                                depth_intrinsics=depth_intrinsics,
                                depth_extrinsics=depth_extrinsics,
                                depth_valid_mask=depth_valid_mask,
                                ee_pos=depth_ee_pos,
                            )
                        else:
                            corrupt_depth_context = encode_depth_context(
                                depth_encoder=depth_encoder,
                                depth_values=depth_values,
                                depth_intrinsics=depth_intrinsics,
                                depth_extrinsics=depth_extrinsics,
                                depth_valid_mask=depth_valid_mask,
                                depth_ee_pos=depth_ee_pos,
                            )
                            corrupt_depth_point_features = compute_depth_point_features(
                                depth_encoder=depth_encoder,
                                depth_values=depth_values,
                                depth_intrinsics=depth_intrinsics,
                                depth_extrinsics=depth_extrinsics,
                                depth_valid_mask=depth_valid_mask,
                                depth_ee_pos=depth_ee_pos,
                            )
                finally:
                    depth_encoder_module.ablation_mode = old_mode
                if depth_fusion_mode == "action_summary_aux":
                    corrupt_depth_point_features = None
                predicted_actions_corrupt = action_head_module.predict_action(
                    actions_hidden_states,
                    depth_context=corrupt_depth_context,
                    query_context=prompt_context,
                    depth_point_features=corrupt_depth_point_features,
                )
            return predicted_actions_corrupt

        if use_contrastive:
            predicted_actions_normal_full = get_normal_full_actions()
            predicted_actions_null = get_null_actions()
            action_loss_normal_full = torch.nn.functional.l1_loss(ground_truth_actions, predicted_actions_normal_full)
            action_loss_null = torch.nn.functional.l1_loss(ground_truth_actions, predicted_actions_null)
            contrastive_loss_null = torch.relu(action_loss_normal_full - action_loss_null + float(contrastive_margin))
            contrastive_loss = contrastive_loss_null
            if depth_encoder is not None:
                predicted_actions_corrupt = get_corrupt_actions()
                action_loss_corrupt = torch.nn.functional.l1_loss(ground_truth_actions, predicted_actions_corrupt)
                contrastive_loss_corrupt = torch.relu(
                    action_loss_normal_full - action_loss_corrupt + float(contrastive_margin)
                )
                contrastive_loss = contrastive_loss + contrastive_loss_corrupt
            loss = loss + float(contrastive_weight) * contrastive_loss

        if float(null_to_base_weight) > 0:
            null_to_base_loss = torch.nn.functional.smooth_l1_loss(get_null_actions(), get_base_actions().detach())
            loss = loss + float(null_to_base_weight) * null_to_base_loss

        if float(corrupt_to_base_weight) > 0 and depth_encoder is not None:
            predicted_actions_corrupt = get_corrupt_actions()
            if action_loss_corrupt is None:
                action_loss_corrupt = torch.nn.functional.l1_loss(ground_truth_actions, predicted_actions_corrupt)
            corrupt_to_base_loss = torch.nn.functional.smooth_l1_loss(
                predicted_actions_corrupt, get_base_actions().detach()
            )
            loss = loss + float(corrupt_to_base_weight) * corrupt_to_base_loss

    spatial_aux_loss = None
    spatial_pred = None
    spatial_target = None
    aux_target = str(aux_target or "none")
    if (
        depth_context is not None
        and depth_aux_spatial_loss_weight > 0
        and depth_fusion_mode in ("action_summary_aux", "object_query")
        and aux_target != "none"
    ):
        aux_depth_context = depth_context_full if depth_context_full is not None else depth_context
        spatial_pred = action_head_module.predict_spatial_delta(
            aux_depth_context,
            actions_hidden_states=actions_hidden_states,
            query_context=prompt_context,
            aux_target=aux_target,
            depth_point_features=depth_point_features_full,
        )
        if aux_target == "next_action_xyz":
            spatial_pred = spatial_pred.to(ground_truth_actions.dtype)
            spatial_target = ground_truth_actions[:, 0, :3]
            if spatial_pred.shape[-1] != spatial_target.shape[-1]:
                raise ValueError(
                    f"aux_target={aux_target!r} requires aux_output_dim={spatial_target.shape[-1]}, "
                    f"got spatial_pred shape {tuple(spatial_pred.shape)}"
                )
            spatial_aux_loss = torch.nn.functional.smooth_l1_loss(spatial_pred, spatial_target)
        elif aux_target in (
            "relative_xyz",
            "contact_xyz",
            "visible_object_rel_xyz",
            "visible_object_point_xyz",
            "ee_to_object_xyz",
            "object_to_target_xyz",
            "gripper_to_contact_distance",
            "task_3d",
            "absolute_keypose",
            "rlbench_keypose_action",
            "point_keypose_xyz",
            "first_close_pose_xyz",
            "pre_first_close_pose_xyz",
            "visible_first_close_point_xyz",
            "visible_pre_first_close_point_xyz",
            "visible_object_point_xyz",
            "future_pose_xyz",
            "final_pose_xyz",
            "farthest_future_pose_xyz",
            "projected_keypose_uv",
        ):
            if "aux_label" not in batch:
                raise KeyError(f"aux_target={aux_target!r} requires batch['aux_label']")
            spatial_pred = spatial_pred.to(ground_truth_actions.dtype)
            spatial_target = batch["aux_label"].to(device_id).to(ground_truth_actions.dtype)
            if spatial_pred.shape[-1] != spatial_target.shape[-1]:
                raise ValueError(
                    f"aux_target={aux_target!r} requires aux_output_dim={spatial_target.shape[-1]}, "
                    f"got spatial_pred shape {tuple(spatial_pred.shape)}"
                )
            spatial_aux_loss = torch.nn.functional.smooth_l1_loss(spatial_pred, spatial_target)
        elif aux_target == "projected_keypose_heatmap":
            if "aux_label" not in batch:
                raise KeyError(f"aux_target={aux_target!r} requires batch['aux_label']")
            spatial_target = batch["aux_label"].to(device_id).to(ground_truth_actions.dtype)
            spatial_pred = spatial_pred.to(ground_truth_actions.dtype)
            if tuple(spatial_pred.shape) != tuple(spatial_target.shape):
                raise ValueError(
                    f"aux_target={aux_target!r} expected prediction shape {tuple(spatial_target.shape)}, "
                    f"got {tuple(spatial_pred.shape)}"
                )
            spatial_aux_loss = torch.nn.functional.mse_loss(spatial_pred.float().sigmoid(), spatial_target.float())
        elif aux_target in ("distance_bin", "relative_z_bin"):
            if "aux_label" not in batch:
                raise KeyError(f"aux_target={aux_target!r} requires batch['aux_label']")
            spatial_target = batch["aux_label"].to(device_id).long()
            spatial_aux_loss = torch.nn.functional.cross_entropy(spatial_pred.float(), spatial_target)
        else:
            raise ValueError(f"Unknown aux_target: {aux_target}")
        loss = loss + float(depth_aux_spatial_loss_weight) * spatial_aux_loss

    curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_actions[:, 0], predicted_actions[:, 0])
    next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_actions[:, 1:], predicted_actions[:, 1:])
    curr_action_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
    curr_action_token_l1 = compute_actions_l1_loss(
        action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
    )

    metrics = {
        "loss_value": loss.item(),
        "action_l1_loss": action_loss.item(),
        "curr_action_accuracy": curr_action_accuracy.item(),
        "curr_action_token_l1_loss": curr_action_token_l1.item(),
        "curr_action_l1_loss": curr_action_l1_loss.item(),
        "next_actions_l1_loss": next_actions_l1_loss.item(),
    }
    if depth_keep_rate is not None:
        metrics["depth_keep_rate"] = depth_keep_rate.item()
    if contrastive_loss is not None:
        metrics["contrastive_loss"] = contrastive_loss.item()
        metrics["contrastive_weighted_loss"] = (float(contrastive_weight) * contrastive_loss).item()
        metrics["action_l1_loss_normal_full"] = action_loss_normal_full.item()
        metrics["action_l1_loss_null"] = action_loss_null.item()
        if contrastive_loss_null is not None:
            metrics["contrastive_loss_null"] = contrastive_loss_null.item()
        if contrastive_loss_corrupt is not None:
            metrics["contrastive_loss_corrupt"] = contrastive_loss_corrupt.item()
        if action_loss_corrupt is not None:
            metrics["action_l1_loss_corrupt"] = action_loss_corrupt.item()
    if null_to_base_loss is not None:
        metrics["null_to_base_loss"] = null_to_base_loss.item()
        metrics["null_to_base_weighted_loss"] = (float(null_to_base_weight) * null_to_base_loss).item()
    if corrupt_to_base_loss is not None:
        metrics["corrupt_to_base_loss"] = corrupt_to_base_loss.item()
        metrics["corrupt_to_base_weighted_loss"] = (float(corrupt_to_base_weight) * corrupt_to_base_loss).item()
        metrics["action_l1_loss_corrupt"] = action_loss_corrupt.item()
    if spatial_aux_loss is not None:
        metrics["spatial_aux_loss"] = spatial_aux_loss.item()
        metrics["aux_weighted_loss"] = (float(depth_aux_spatial_loss_weight) * spatial_aux_loss).item()
    if hasattr(action_head_module, "depth_fusion_gate"):
        metrics["depth_fusion_gate"] = action_head_module.depth_fusion_gate.detach().float().item()
    if depth_encoder is not None:
        metrics["depth_alpha"] = unwrap_module(depth_encoder).alpha.detach().float().item()
    if keypose_action_residual is not None:
        metrics["keypose_action_residual_abs_mean"] = keypose_action_residual.detach().float().abs().mean().item()
    if point_action_residual is not None:
        metrics["point_action_residual_abs_mean"] = point_action_residual.detach().float().abs().mean().item()
    if log_diagnostics:
        print(f"DepthVLA main action L1 loss: {action_loss.detach().float().item():.6g}")
        if depth_keep_rate is not None:
            print(f"DepthVLA depth dropout: {float(depth_dropout):.6g}; keep rate this batch: {depth_keep_rate.detach().float().item():.6g}")
        if contrastive_loss is not None:
            print(f"DepthVLA contrastive margin: {float(contrastive_margin):.6g}")
            print(f"DepthVLA contrastive weight: {float(contrastive_weight):.6g}")
            print(f"DepthVLA normal-full action L1 loss: {action_loss_normal_full.detach().float().item():.6g}")
            print(f"DepthVLA null action L1 loss: {action_loss_null.detach().float().item():.6g}")
            if action_loss_corrupt is not None:
                print(f"DepthVLA corrupt depth mode: {str(corrupt_depth_mode or 'shuffle_depth')}")
                print(f"DepthVLA corrupt action L1 loss: {action_loss_corrupt.detach().float().item():.6g}")
            if contrastive_loss_null is not None:
                print(f"DepthVLA contrastive null loss: {contrastive_loss_null.detach().float().item():.6g}")
            if contrastive_loss_corrupt is not None:
                print(f"DepthVLA contrastive corrupt loss: {contrastive_loss_corrupt.detach().float().item():.6g}")
            print(f"DepthVLA contrastive loss: {contrastive_loss.detach().float().item():.6g}")
        if null_to_base_loss is not None:
            print(f"DepthVLA null-to-base weight: {float(null_to_base_weight):.6g}")
            print(f"DepthVLA null-to-base loss: {null_to_base_loss.detach().float().item():.6g}")
        if corrupt_to_base_loss is not None:
            print(f"DepthVLA corrupt depth mode: {str(corrupt_depth_mode or 'shuffle_depth')}")
            print(f"DepthVLA corrupt-to-base weight: {float(corrupt_to_base_weight):.6g}")
            print(f"DepthVLA corrupt action L1 loss: {action_loss_corrupt.detach().float().item():.6g}")
            print(f"DepthVLA corrupt-to-base loss: {corrupt_to_base_loss.detach().float().item():.6g}")
        if spatial_aux_loss is not None:
            print(f"DepthVLA auxiliary target: {aux_target}")
            print(f"DepthVLA auxiliary prediction shape: {tuple(spatial_pred.shape)}")
            print(f"DepthVLA auxiliary label shape: {tuple(spatial_target.shape)}")
            print(f"DepthVLA auxiliary spatial loss: {spatial_aux_loss.detach().float().item():.6g}")
            print(f"DepthVLA auxiliary loss weight: {float(depth_aux_spatial_loss_weight):.6g}")
    return loss, metrics


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    action_head,
    depth_encoder,
    train_dataset,
    distributed_state,
) -> None:
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"
    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        unwrap_module(vla).save_pretrained(adapter_dir)
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(unwrap_module(proprio_projector).state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")
        torch.save(unwrap_module(action_head).state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")
        if cfg.use_depth and depth_encoder is not None:
            torch.save(unwrap_module(depth_encoder).state_dict(), checkpoint_dir / f"depth_encoder--{checkpoint_name_suffix}")
            src_stats = run_dir / "geometry_norm_stats.json"
            if src_stats.exists():
                with open(src_stats, "r") as f:
                    save_geometry_norm_stats(json.load(f), checkpoint_dir / "geometry_norm_stats.json")
        print(f"Saved DepthVLA checkpoint for Step {log_step} at: {checkpoint_dir}")

    distributed_barrier()

    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir).merge_and_unload()
        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")
        distributed_barrier()


@draccus.wrap()
def finetune(cfg: DepthFinetuneConfig) -> None:
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert cfg.num_images_in_input == 2, "DepthVLA-OFT v1 expects agentview + wrist RGB inputs."
    assert not cfg.image_aug, "DepthVLA-OFT v1 keeps image_aug=False to avoid RGB/depth misalignment."
    depth_integration_mode = apply_depth_integration_mode(cfg)
    apply_v1_summary_config(cfg)

    cfg.vla_path = cfg.vla_path.rstrip("/")
    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        print(f"DepthVLA integration mode: {depth_integration_mode}")
        print(f"DepthVLA use_depth: {cfg.use_depth}")
        print(f"DepthVLA depth_encoder_type: {cfg.depth_encoder_type}")
        if cfg.depth_encoder_type == "dense_point":
            print(f"DepthVLA depth_num_points_per_view: {cfg.depth_num_points_per_view}")
        print(f"DepthVLA depth_fusion_mode: {cfg.depth_fusion_mode}")
        print(f"DepthVLA action fusion gate init: {cfg.depth_action_fusion_gate_init}")
        print(f"DepthVLA appends depth tokens to prefix: {cfg.use_depth and cfg.depth_fusion_mode == 'prefix'}")

    if cfg.use_wandb and distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    if model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)
    distributed_barrier()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device_id)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    resume_component_path = cfg.resume_components_from or (cfg.vla_path if cfg.resume else None)
    if resume_component_path is not None:
        adapter_dir = Path(resume_component_path) / "lora_adapter"
        if adapter_dir.exists():
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file as load_safetensors

            adapter_path = adapter_dir / "adapter_model.safetensors"
            if distributed_state.is_main_process:
                print(f"Loading LoRA adapter weights from: {adapter_path}")
            set_peft_model_state_dict(vla, load_safetensors(str(adapter_path), device="cpu"))
    if cfg.freeze_vla_lora:
        freeze_module(vla)
    vla.print_trainable_parameters()
    vla = wrap_ddp(vla, device_id, find_unused=True)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = ProprioProjector(llm_dim=unwrap_module(vla).llm_dim, proprio_dim=PROPRIO_DIM)
        count_parameters(proprio_projector, "proprio_projector")
        if resume_component_path is not None:
            proprio_projector.load_state_dict(load_checkpoint("proprio_projector", resume_component_path, cfg.resume_step))
        if cfg.freeze_proprio_projector:
            freeze_module(proprio_projector)
        proprio_projector = wrap_ddp(proprio_projector.to(device_id), device_id)

    use_action_head_depth_fusion = cfg.use_depth and cfg.depth_fusion_mode in ("action_head", "action_residual", "action_summary_aux", "object_query")
    if cfg.depth_fusion_mode == "action_residual":
        depth_fusion_type = "action_residual"
    elif cfg.depth_fusion_mode == "action_summary_aux":
        depth_fusion_type = "action_summary_aux"
    elif cfg.depth_fusion_mode == "object_query":
        depth_fusion_type = "object_query"
    else:
        depth_fusion_type = "hidden_film"
    if cfg.aux_target == "projected_keypose_heatmap":
        expected_aux_dim = 2 * int(cfg.aux_heatmap_size) * int(cfg.aux_heatmap_size)
        if int(cfg.aux_output_dim) != expected_aux_dim:
            raise ValueError(
                "aux_target='projected_keypose_heatmap' requires "
                f"--aux_output_dim {expected_aux_dim} for aux_heatmap_size={cfg.aux_heatmap_size}, "
                f"got {cfg.aux_output_dim}"
            )
    action_head = L1RegressionActionHead(
        input_dim=unwrap_module(vla).llm_dim,
        hidden_dim=unwrap_module(vla).llm_dim,
        action_dim=ACTION_DIM,
        use_depth_conditioning=use_action_head_depth_fusion,
        depth_fusion_type=depth_fusion_type,
        depth_fusion_gate_init=cfg.depth_action_fusion_gate_init,
        depth_hidden_delta_clip=cfg.depth_hidden_delta_clip,
        depth_action_residual_clip=cfg.depth_action_residual_clip,
        depth_keypose_residual_weight=cfg.depth_keypose_residual_weight,
        depth_keypose_residual_clip=cfg.depth_keypose_residual_clip,
        depth_point_action_weight=cfg.depth_point_action_weight,
        depth_point_action_clip=cfg.depth_point_action_clip,
        depth_waypoint_action_weight=cfg.depth_waypoint_action_weight,
        depth_waypoint_action_clip=cfg.depth_waypoint_action_clip,
        depth_waypoint_action_scale=cfg.depth_waypoint_action_scale,
        depth_waypoint_action_chunk_len=cfg.depth_waypoint_action_chunk_len,
        depth_adapter_hidden_dim=cfg.depth_adapter_hidden_dim,
        spatial_aux_output_dim=cfg.aux_output_dim,
    )
    count_parameters(action_head, "action_head")
    if resume_component_path is not None:
        action_state = load_checkpoint("action_head", resume_component_path, cfg.resume_step)
        incompatible = action_head.load_state_dict(action_state, strict=not use_action_head_depth_fusion)
        if use_action_head_depth_fusion and distributed_state.is_main_process:
            print(
                "Loaded action head with strict=False for depth adapter; "
                f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
            )
    if cfg.freeze_action_head_base:
        freeze_action_head_base(action_head)
    action_head = wrap_ddp(action_head.to(torch.bfloat16).to(device_id), device_id)

    depth_encoder = None
    if cfg.use_depth:
        if cfg.depth_encoder_type == "grid":
            depth_encoder = LightweightDepthTokenEncoder(
                llm_dim=unwrap_module(vla).llm_dim,
                hidden_dim=cfg.depth_hidden_dim,
                grid_size=cfg.depth_grid_size,
                depth_min_m=cfg.depth_min_m,
                depth_max_m=cfg.depth_max_m,
                num_views=2,
                geometry_norm=cfg.geometry_norm,
                geometry_clip=cfg.geometry_clip,
                enable_summary=cfg.depth_fusion_mode == "action_summary_aux",
                summary_repr=cfg.summary_repr,
                summary_pool=cfg.summary_pool,
            )
        elif cfg.depth_encoder_type == "dense_point":
            if cfg.geometry_norm != "none":
                raise ValueError("depth_encoder_type='dense_point' currently requires --geometry_norm none")
            if cfg.depth_fusion_mode == "prefix":
                raise ValueError("depth_encoder_type='dense_point' is intended for action-side/object-query fusion, not prefix append")
            depth_encoder = DensePointDepthTokenEncoder(
                llm_dim=unwrap_module(vla).llm_dim,
                hidden_dim=cfg.depth_hidden_dim,
                num_points_per_view=cfg.depth_num_points_per_view,
                depth_min_m=cfg.depth_min_m,
                depth_max_m=cfg.depth_max_m,
                num_views=2,
            )
        else:
            raise ValueError("Unknown depth_encoder_type; choose from grid|dense_point")
        if cfg.resume and cfg.resume_components_from is None:
            depth_encoder.load_state_dict(load_checkpoint("depth_encoder", cfg.vla_path, cfg.resume_step))
        configure_depth_alpha(depth_encoder, cfg.depth_alpha_init, cfg.freeze_depth_alpha, cfg.min_depth_alpha)
        count_parameters(depth_encoder, "depth_encoder")
        depth_encoder = wrap_ddp(depth_encoder.to(torch.bfloat16).to(device_id), device_id)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder

    train_dataset = LiberoRGBDHDF5Dataset(
        cfg.rgbd_data_dir,
        cfg.dataset_name,
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_depth=cfg.use_depth,
        use_proprio=cfg.use_proprio,
        aux_target=cfg.aux_target if cfg.depth_fusion_mode in ("action_summary_aux", "object_query") and cfg.depth_aux_spatial_loss_weight > 0 else "none",
        aux_heatmap_size=cfg.aux_heatmap_size,
        aux_heatmap_sigma=cfg.aux_heatmap_sigma,
        aux_future_horizon=cfg.aux_future_horizon,
        aux_distance_bin_edges=cfg.aux_distance_bin_edges,
        aux_z_bin_edges=cfg.aux_z_bin_edges,
    )
    use_depth = cfg.use_depth and depth_encoder is not None
    geometry_norm_stats = None
    if use_depth and cfg.geometry_norm == "dataset_std":
        geometry_norm_stats = compute_geometry_norm_stats(train_dataset, depth_encoder, cfg.geometry_clip)
        unwrap_module(depth_encoder).set_geometry_normalization(geometry_norm_stats, cfg.geometry_norm, cfg.geometry_clip)
    elif use_depth and cfg.geometry_norm != "none":
        raise ValueError(f"Unknown geometry_norm mode: {cfg.geometry_norm}")

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
        if geometry_norm_stats is not None:
            save_geometry_norm_stats(geometry_norm_stats, run_dir / "geometry_norm_stats.json")
        with open(run_dir / "depthvla_config.json", "w") as f:
            json.dump(
                {
                    "depth_integration_mode": cfg.depth_integration_mode,
                    "use_depth": cfg.use_depth,
                    "depth_encoder_type": cfg.depth_encoder_type,
                    "depth_grid_size": cfg.depth_grid_size,
                    "depth_num_points_per_view": cfg.depth_num_points_per_view,
                    "depth_fusion_mode": cfg.depth_fusion_mode,
                    "depth_action_fusion_gate_init": cfg.depth_action_fusion_gate_init,
                    "depth_hidden_delta_clip": cfg.depth_hidden_delta_clip,
                    "depth_action_residual_clip": cfg.depth_action_residual_clip,
                    "depth_keypose_residual_weight": cfg.depth_keypose_residual_weight,
                    "depth_keypose_residual_clip": cfg.depth_keypose_residual_clip,
                    "depth_point_action_weight": cfg.depth_point_action_weight,
                    "depth_point_action_clip": cfg.depth_point_action_clip,
                    "depth_waypoint_action_weight": cfg.depth_waypoint_action_weight,
                    "depth_waypoint_action_clip": cfg.depth_waypoint_action_clip,
                    "depth_waypoint_action_scale": cfg.depth_waypoint_action_scale,
                    "depth_waypoint_action_chunk_len": cfg.depth_waypoint_action_chunk_len,
                    "depth_adapter_hidden_dim": cfg.depth_adapter_hidden_dim,
                    "summary_repr": cfg.summary_repr,
                    "summary_pool": cfg.summary_pool,
                    "depth_aux_spatial_loss_weight": cfg.depth_aux_spatial_loss_weight,
                    "aux_target": cfg.aux_target,
                    "aux_output_dim": cfg.aux_output_dim,
                    "aux_future_horizon": cfg.aux_future_horizon,
                    "aux_heatmap_size": cfg.aux_heatmap_size,
                    "aux_heatmap_sigma": cfg.aux_heatmap_sigma,
                    "aux_distance_bin_edges": cfg.aux_distance_bin_edges,
                    "aux_z_bin_edges": cfg.aux_z_bin_edges,
                    "freeze_vla_lora": cfg.freeze_vla_lora,
                    "freeze_proprio_projector": cfg.freeze_proprio_projector,
                    "freeze_action_head_base": cfg.freeze_action_head_base,
                    "resume_components_from": str(cfg.resume_components_from) if cfg.resume_components_from else None,
                    "resume_step": cfg.resume_step,
                    "depth_alpha_init": cfg.depth_alpha_init,
                    "freeze_depth_alpha": cfg.freeze_depth_alpha,
                    "min_depth_alpha": cfg.min_depth_alpha,
                    "depth_dropout": cfg.depth_dropout,
                    "use_contrastive": cfg.use_contrastive,
                    "contrastive_weight": cfg.contrastive_weight,
                    "contrastive_margin": cfg.contrastive_margin,
                    "null_to_base_weight": cfg.null_to_base_weight,
                    "corrupt_to_base_weight": cfg.corrupt_to_base_weight,
                    "corrupt_depth_mode": cfg.corrupt_depth_mode,
                    "geometry_norm": cfg.geometry_norm,
                    "geometry_clip": cfg.geometry_clip,
                },
                f,
                indent=2,
            )

    collator = DepthPaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator, num_workers=2)

    use_depth_prefix = use_depth and cfg.depth_fusion_mode == "prefix"
    depth_num_tokens = unwrap_module(depth_encoder).depth_num_tokens if use_depth_prefix else 0
    NUM_PATCHES = unwrap_module(vla).get_num_prefix_tokens(
        use_depth=use_depth_prefix,
        depth_num_tokens=depth_num_tokens,
        use_proprio=cfg.use_proprio,
        use_diffusion=False,
    )
    print(f"DepthVLA prefix tokens: {NUM_PATCHES}")
    print(f"DepthVLA depth token count included in prefix: {depth_num_tokens}")

    if distributed_state.is_main_process:
        summarize_trainable_parameters(vla, "vla")
        summarize_trainable_parameters(action_head, "action_head")
        if proprio_projector is not None:
            summarize_trainable_parameters(proprio_projector, "proprio_projector")
        if depth_encoder is not None:
            summarize_trainable_parameters(depth_encoder, "depth_encoder")

    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    trainable_params += [p for p in action_head.parameters() if p.requires_grad]
    if cfg.use_proprio:
        trainable_params += [p for p in proprio_projector.parameters() if p.requires_grad]
    if use_depth:
        trainable_params += [p for p in depth_encoder.parameters() if p.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")

    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_token_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "spatial_aux_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "aux_weighted_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "depth_fusion_gate": deque(maxlen=cfg.grad_accumulation_steps),
        "depth_alpha": deque(maxlen=cfg.grad_accumulation_steps),
        "depth_keep_rate": deque(maxlen=cfg.grad_accumulation_steps),
        "contrastive_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "contrastive_loss_null": deque(maxlen=cfg.grad_accumulation_steps),
        "contrastive_loss_corrupt": deque(maxlen=cfg.grad_accumulation_steps),
        "contrastive_weighted_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "action_l1_loss_normal_full": deque(maxlen=cfg.grad_accumulation_steps),
        "action_l1_loss_null": deque(maxlen=cfg.grad_accumulation_steps),
        "null_to_base_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "null_to_base_weighted_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "corrupt_to_base_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "corrupt_to_base_weighted_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "action_l1_loss_corrupt": deque(maxlen=cfg.grad_accumulation_steps),
        "keypose_action_residual_abs_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "point_action_residual_abs_mean": deque(maxlen=cfg.grad_accumulation_steps),
    }

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        action_head.train()
        if proprio_projector is not None:
            proprio_projector.train()
        if depth_encoder is not None:
            depth_encoder.train()
        optimizer.zero_grad()

        batch_idx = 0
        while True:
            for batch in dataloader:
                loss, metrics = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    depth_encoder=depth_encoder if use_depth else None,
                    batch=batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_proprio=cfg.use_proprio,
                    use_depth=use_depth,
                    use_film=False,
                    num_patches=NUM_PATCHES,
                    depth_fusion_mode=cfg.depth_fusion_mode,
                    depth_aux_spatial_loss_weight=cfg.depth_aux_spatial_loss_weight,
                    depth_integration_mode=cfg.depth_integration_mode,
                    aux_target=cfg.aux_target,
                    depth_dropout=cfg.depth_dropout,
                    use_contrastive=cfg.use_contrastive,
                    contrastive_weight=cfg.contrastive_weight,
                    contrastive_margin=cfg.contrastive_margin,
                    null_to_base_weight=cfg.null_to_base_weight,
                    corrupt_to_base_weight=cfg.corrupt_to_base_weight,
                    corrupt_depth_mode=cfg.corrupt_depth_mode,
                    log_diagnostics=distributed_state.is_main_process and batch_idx == 0,
                )
                (loss / cfg.grad_accumulation_steps).backward()

                for metric_name, value in metrics.items():
                    if metric_name in recent_metrics:
                        recent_metrics[metric_name].append(value)

                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
                log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
                if cfg.use_wandb and distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                    log_metrics_to_wandb(compute_smoothened_metrics(recent_metrics), "DepthVLA Train", log_step, wandb)
                    wandb.log({"DepthVLA Train/Learning Rate": scheduler.get_last_lr()[0]}, step=log_step)

                if cfg.lr_warmup_steps > 0:
                    lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = original_lr * (0.1 + 0.9 * lr_progress)

                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    clamp_depth_alpha(depth_encoder if use_depth else None, cfg.min_depth_alpha)
                    scheduler.step()
                    optimizer.zero_grad()
                    progress.update()

                if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                    save_training_checkpoint(
                        cfg,
                        run_dir,
                        log_step,
                        vla,
                        processor,
                        proprio_projector if cfg.use_proprio else None,
                        action_head,
                        depth_encoder if use_depth else None,
                        train_dataset,
                        distributed_state,
                    )

                if log_step >= cfg.max_steps:
                    return
                batch_idx += 1


if __name__ == "__main__":
    finetune()
