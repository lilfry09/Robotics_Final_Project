# 2026-07-04 Reach Safe RGB-Anchor Run

## Goal

Validate whether low-gate, clamped RGB-D residual fusion can protect the successful RGB-only reach policy while preserving a measurable causal effect from depth.

## Setup

- Dataset: `/root/RLBench/rgbd_hdf5_reach_3demos_64`
- Eval data: `/root/RLBench/peract_dataset/stable6_3demos_64`
- Checkpoint root: `/root/runs_rlbench_reach_safe_rgbanchor`
- Resume anchor: successful reach RGB-only checkpoint in `/root/runs_rlbench_reach_3demos/...--rlbench-rgb-only`
- Fusion:
  - dense point tokens, `1024` points per view
  - object-query depth fusion
  - `DEPTH_ACTION_FUSION_GATE_INIT=0.01`
  - `DEPTH_HIDDEN_DELTA_CLIP=0.05`
  - `DEPTH_ACTION_RESIDUAL_CLIP=0.02`
  - frozen VLA LoRA, proprio projector, and action-head base
  - auxiliary target: `absolute_keypose`, weight `0.2`

## Result

Rollout, `reach_target`, `1` episode, horizon `150`, `MAX_DELTA_XYZ=0.05`:

| depth mode | success | length | delta xyz mean |
|---|---:|---:|---:|
| normal | `1/1` | `29` | `0.0122476` |
| null | `1/1` | `29` | `0.0122476` |
| cross_sample | `1/1` | `29` | `0.0122476` |

Offline action diagnostic, `7` samples:

| depth mode | xyz RMSE | xyz direction cosine |
|---|---:|---:|
| normal | `0.001700` | `0.97756` |
| null | `0.001700` | `0.97756` |
| cross_sample | `0.001700` | `0.97756` |

## Interpretation

The RGB-anchor repair fixed the previous harmful behavior: normal depth no longer destroys the successful RGB reach policy. However, this is not depth usage. The action predictions and rollout traces are identical under real, zero, and cross-sample depth.

This is a sanity pass but a causal no-go.

## Decision

- Keep this recipe as a protective baseline.
- Do not keep tuning reach-only residuals for a depth claim.
- Move the next causal gate to stronger 3D/contact tasks and explicit action-space grounding: keypose-conditioned residual, projected heatmap, or 3D action map.

## Cleanup

Deleted obsolete failed `/root/runs_rlbench_stable6_3demos` checkpoints after results were recorded, freeing about `1.5G`.

Deleted obsolete LIBERO-era checkpoint directories:

- `/root/autodl-tmp/openvla-oft/runs_depthvla_plus_rgb_mix_phaseA`
- `/root/autodl-tmp/openvla-oft/runs_depthvla_action_summary_v1`
- `/root/autodl-tmp/openvla-oft/runs_depthvla_stage2`
- `/root/autodl-tmp/openvla-oft/runs_depth_fix`
- `/root/autodl-tmp/openvla-oft/runs_depthvla_object_query_task3d_contrastive_1k`
- `/root/autodl-tmp/openvla-oft/runs_depthvla_object_query_task3d_pilot`
- `/root/autodl-tmp/openvla-oft/runs_quick_heatmap`
- `/root/autodl-tmp/openvla-oft/runs_rlbench_rgbd`

`/root/autodl-tmp` free space improved from about `4.7G` to `13G`.
