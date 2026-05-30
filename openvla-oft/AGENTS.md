# AGENTS.md

Guidance for coding agents working in this repository.

## Project Context

This repo is based on OpenVLA-OFT and is being extended with DepthVLA-OFT:
OpenVLA-OFT plus lightweight metric depth geometry tokens for LIBERO-style robot manipulation.

The intended first implementation path is:
- Keep OpenVLA-OFT's RGB backbone, proprio token, LoRA fine-tuning, and continuous L1 action head.
- Add metric base/world-frame depth geometry tokens through `LightweightDepthTokenEncoder`.
- Train a matched RGB-only baseline and RGB-D model from the same regenerated RGB-D HDF5 demonstrations.
- Treat official OpenVLA-OFT checkpoints as references, not as the matched baseline for the RGB-D pipeline.

## Important Files

- `prismatic/extern/hf/modeling_prismatic.py`: HuggingFace OpenVLA/OFT model logic. Depth tokens are appended here.
- `prismatic/models/depth_encoder.py`: Lightweight depth-to-token encoder.
- `vla-scripts/finetune.py`: Original OpenVLA-OFT RLDS fine-tuning script.
- `vla-scripts/finetune_depthvla.py`: DepthVLA-OFT HDF5 fine-tuning script.
- `experiments/robot/libero/regenerate_libero_dataset.py`: Original LIBERO HDF5 regeneration script.
- `experiments/robot/libero/regenerate_libero_rgbd_dataset.py`: RGB-D regeneration path for DepthVLA-OFT.
- `experiments/robot/libero/run_libero_eval.py`: LIBERO evaluation entrypoint.
- `experiments/robot/openvla_utils.py`: OpenVLA loading and inference helpers.
- `tests/test_depth_encoder.py`: Focused tests for the depth encoder.

## DepthVLA-OFT Invariants

- Depth from MuJoCo / robosuite must be converted from normalized `[0, 1]` depth to metric depth using robosuite camera utilities before being saved or used.
- Multi-view geometry must use camera extrinsics. Do not combine `agentview` and wrist-camera XYZ in unrelated camera-local frames.
- HDF5 should store raw, unrotated RGB and depth. The dataset/inference path rotates RGB by 180 degrees for OpenVLA compatibility, and flips the coarse geometry-token grid to preserve alignment.
- Do not back-project rotated depth with unmodified intrinsics.
- Depth token count must be included in prefix-length / action-hidden-state slicing in both training and inference.
- For v1, keep `image_aug=False` for both RGB-only and RGB-D HDF5 runs unless paired RGB-depth geometric augmentation and intrinsics updates are implemented.

## Development Rules

- Prefer small, compatible changes over broad rewrites.
- Do not break the existing `use_depth=False` OpenVLA-OFT path.
- Follow the existing projector/component pattern: instantiate auxiliary modules in scripts, pass them into model forward/predict calls, and save them as separate checkpoint files.
- Use `rg` for search and inspect surrounding code before editing.
- Use `apply_patch` for manual edits.
- Do not revert user changes or unrelated dirty files.
- Keep comments sparse and useful.

## Verification

Before handing off changes, run at least:

```bash
python -m py_compile prismatic/models/depth_encoder.py prismatic/extern/hf/modeling_prismatic.py
python tests/test_depth_encoder.py
```

When LIBERO / robosuite are installed, additionally run:

```bash
python experiments/robot/libero/regenerate_libero_rgbd_dataset.py --help
python experiments/robot/libero/run_libero_eval.py --help
```

For real experiments, smoke-test in this order:
- Generate one small RGB-D HDF5 task subset.
- Run one-step or tiny-step `finetune_depthvla.py` with `--use_depth False`.
- Run the same tiny training with `--use_depth True --depth_grid_size 4`.
- Run one LIBERO rollout with `--use_depth True`.

## Reporting Results

When writing reports or experiment notes:
- Call the fair baseline "matched RGB-only OpenVLA-OFT trained on the regenerated HDF5 pipeline."
- Do not claim reproduction of official OpenVLA-OFT leaderboard results unless that has been directly verified.
- Primary comparison should be matched RGB-only vs DepthVLA-OFT under identical data, seeds, steps, batch size, LR, proprio setting, and action head.
