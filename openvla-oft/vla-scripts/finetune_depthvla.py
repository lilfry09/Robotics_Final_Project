"""
finetune_depthvla.py

Fine-tunes OpenVLA-OFT on regenerated LIBERO RGB-D HDF5 demonstrations.
Set --use_depth False to train the matched RGB-only baseline on the same data.
"""

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
    use_depth: bool = True
    depth_grid_size: int = 4
    depth_hidden_dim: int = 256
    depth_min_m: float = 0.01
    depth_max_m: float = 5.0
    geometry_norm: str = "none"
    geometry_clip: float = 5.0

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
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    return remove_ddp_in_checkpoint(torch.load(checkpoint_path, weights_only=True, map_location=device))


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def distributed_barrier() -> None:
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        distributed_barrier()


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def get_run_id(cfg: DepthFinetuneConfig) -> str:
    if cfg.run_id_override is not None:
        return cfg.run_id_override
    depth_tag = f"depth-g{cfg.depth_grid_size}" if cfg.use_depth else "rgb-only"
    if cfg.use_depth and cfg.geometry_norm != "none":
        depth_tag += f"+geom-{cfg.geometry_norm}+clip-{cfg.geometry_clip}"
    run_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}+{depth_tag}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def standardize_libero_actions(actions: np.ndarray) -> np.ndarray:
    actions = actions.astype(np.float32).copy()
    gripper = 1.0 - np.clip(actions[:, -1:], 0.0, 1.0)
    return np.concatenate([actions[:, :6], gripper], axis=1).astype(np.float32)


def get_libero_proprio(obs_group) -> np.ndarray:
    ee_states = obs_group["ee_states"][()].astype(np.float32)
    gripper_states = obs_group["gripper_states"][()].astype(np.float32)
    return np.concatenate([ee_states, gripper_states], axis=1).astype(np.float32)


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
    normalized = np.where(low == high, 0.0, normalized)
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
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.use_depth = use_depth
        self.use_proprio = use_proprio
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
            return value.decode() if isinstance(value, bytes) else str(value)
        if "language_instruction" in demo:
            value = demo["language_instruction"][()]
            return value.decode() if isinstance(value, bytes) else str(value)
        return file_path.stem.replace("_demo", "").replace("_", " ")

    def _load_episode(self, file_path: str, demo_key: str) -> Dict:
        cache_key = (file_path, demo_key)
        if cache_key in self.episode_cache:
            return self.episode_cache[cache_key]
        with h5py.File(file_path, "r") as f:
            demo = f["data"][demo_key]
            obs = demo["obs"]
            ep = {
                "actions": standardize_libero_actions(demo["actions"][()]),
                "proprio": get_libero_proprio(obs),
                "agentview_rgb": obs["agentview_rgb"][()],
                "eye_in_hand_rgb": obs["eye_in_hand_rgb"][()],
            }
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

        rgb = Image.fromarray(ep["agentview_rgb"][t][::-1, ::-1]).convert("RGB")
        wrist_rgb = Image.fromarray(ep["eye_in_hand_rgb"][t][::-1, ::-1]).convert("RGB")
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
            for key in ("depth_values", "depth_intrinsics", "depth_extrinsics", "depth_valid_mask"):
                output[key] = torch.tensor(np.stack([instance[key] for instance in instances]))
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
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    depth_kwargs = {}
    if use_depth:
        depth_kwargs = {
            "depth_values": batch["depth_values"].to(device_id),
            "depth_intrinsics": batch["depth_intrinsics"].to(device_id),
            "depth_extrinsics": batch["depth_extrinsics"].to(device_id),
            "depth_valid_mask": batch["depth_valid_mask"].to(device_id),
            "depth_encoder": depth_encoder,
        }

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
            **depth_kwargs,
        )

    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)

    last_hidden_states = output.hidden_states[-1]
    text_hidden_states = last_hidden_states[:, num_patches:-1]
    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )
    predicted_actions = action_head.module.predict_action(actions_hidden_states)
    loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

    curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_actions[:, 0], predicted_actions[:, 0])
    next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_actions[:, 1:], predicted_actions[:, 1:])
    curr_action_accuracy = compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=current_action_mask)
    curr_action_token_l1 = compute_actions_l1_loss(
        action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
    )

    return loss, {
        "loss_value": loss.item(),
        "curr_action_accuracy": curr_action_accuracy.item(),
        "curr_action_token_l1_loss": curr_action_token_l1.item(),
        "curr_action_l1_loss": curr_action_l1_loss.item(),
        "next_actions_l1_loss": next_actions_l1_loss.item(),
    }


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
        vla.module.save_pretrained(adapter_dir)
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")
        torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")
        if cfg.use_depth and depth_encoder is not None:
            torch.save(depth_encoder.state_dict(), checkpoint_dir / f"depth_encoder--{checkpoint_name_suffix}")
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

    cfg.vla_path = cfg.vla_path.rstrip("/")
    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

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
    vla.print_trainable_parameters()
    vla = wrap_ddp(vla, device_id, find_unused=True)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = ProprioProjector(llm_dim=vla.module.llm_dim, proprio_dim=PROPRIO_DIM)
        count_parameters(proprio_projector, "proprio_projector")
        if cfg.resume:
            proprio_projector.load_state_dict(load_checkpoint("proprio_projector", cfg.vla_path, cfg.resume_step))
        proprio_projector = wrap_ddp(proprio_projector.to(device_id), device_id)

    action_head = L1RegressionActionHead(input_dim=vla.module.llm_dim, hidden_dim=vla.module.llm_dim, action_dim=ACTION_DIM)
    count_parameters(action_head, "action_head")
    if cfg.resume:
        action_head.load_state_dict(load_checkpoint("action_head", cfg.vla_path, cfg.resume_step))
    action_head = wrap_ddp(action_head.to(torch.bfloat16).to(device_id), device_id)

    depth_encoder = None
    if cfg.use_depth:
        depth_encoder = LightweightDepthTokenEncoder(
            llm_dim=vla.module.llm_dim,
            hidden_dim=cfg.depth_hidden_dim,
            grid_size=cfg.depth_grid_size,
            depth_min_m=cfg.depth_min_m,
            depth_max_m=cfg.depth_max_m,
            num_views=2,
            geometry_norm=cfg.geometry_norm,
            geometry_clip=cfg.geometry_clip,
        )
        count_parameters(depth_encoder, "depth_encoder")
        if cfg.resume:
            depth_encoder.load_state_dict(load_checkpoint("depth_encoder", cfg.vla_path, cfg.resume_step))
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
    )
    use_depth = cfg.use_depth and depth_encoder is not None
    geometry_norm_stats = None
    if use_depth and cfg.geometry_norm == "dataset_std":
        geometry_norm_stats = compute_geometry_norm_stats(train_dataset, depth_encoder, cfg.geometry_clip)
        depth_encoder.module.set_geometry_normalization(geometry_norm_stats, cfg.geometry_norm, cfg.geometry_clip)
    elif use_depth and cfg.geometry_norm != "none":
        raise ValueError(f"Unknown geometry_norm mode: {cfg.geometry_norm}")

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
        if geometry_norm_stats is not None:
            save_geometry_norm_stats(geometry_norm_stats, run_dir / "geometry_norm_stats.json")
        with open(run_dir / "depthvla_config.json", "w") as f:
            json.dump(
                {
                    "use_depth": cfg.use_depth,
                    "depth_grid_size": cfg.depth_grid_size,
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

    depth_num_tokens = depth_encoder.module.depth_num_tokens if use_depth else 0
    NUM_PATCHES = vla.module.get_num_prefix_tokens(
        use_depth=use_depth,
        depth_num_tokens=depth_num_tokens,
        use_proprio=cfg.use_proprio,
        use_diffusion=False,
    )
    print(f"DepthVLA prefix tokens: {NUM_PATCHES}")

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
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_token_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
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
