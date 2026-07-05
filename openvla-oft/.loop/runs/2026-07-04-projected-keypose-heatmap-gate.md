# Run: Projected Keypose Heatmap Gate

Date: 2026-07-04 UTC

## Question

Can a BridgeVLA-style projected 2D heatmap create a real normal-vs-corrupt depth separation before training another 7B policy checkpoint?

## Implementation

Added:

```text
experiments/robot/rlbench/probe_projected_keypose_heatmap.py
```

Runner command:

```bash
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh projected-heatmap-probe
```

The probe trains a small CNN on normal depth-derived geometry maps and predicts where `rlbench_keypose_action[:3]` projects into `agentview` and `eye_in_hand`. It evaluates the same model with normal, null, and cross-sample inputs.

## Results

`open_drawer`, `3 demos / 317 transitions`:

```text
normal peak error:       2.85 px
cross-sample peak error: 8.86 px
null peak error:         43.30 px
normal-vs-cross delta:   8.11 px
```

stable6, `18 demos / 2009 transitions`:

```text
normal peak error:       2.94 px
cross-sample peak error: 12.53 px
null peak error:         57.63 px
normal-vs-cross delta:   12.13 px
```

## Decision

GO for implementation as the next spatial-action bottleneck.

Do not claim RGB-D beats RGB yet. This only proves that the projected heatmap target is causally sensitive to real depth. Next, the heatmap output must be wired into final action formation and must produce nonzero paired normal-vs-cross action deltas before rollout.

## Follow-Up Implementation

Added a minimal trainable bottleneck:

```text
aux_target = projected_keypose_uv
aux_output_dim = 4
```

This target projects `rlbench_keypose_action[:3]` into `agentview` and `eye_in_hand` as normalized UV coordinates. It reuses the object-query spatial head and bounded keypose-residual action path.

Verification:

- Synthetic HDF5 dataset smoke passed.
- Real `open_drawer` `MAX_STEPS=1` training smoke passed.
- Aux prediction shape: `(1, 4)`.
- Aux label shape: `(1, 4)`.

Next gate:

Train `open_drawer` projected-UV residual from the RGB-only anchor. Before rollout, run paired normal-vs-cross diagnostic and reject the run if action deltas are still zero.

## Projected-UV Policy Gate

Formal run:

```text
task: open_drawer
steps: 5000
aux_target: projected_keypose_uv
aux_output_dim: 4
resume anchor: RGB-only open_drawer checkpoint
frozen modules: RGB/VLA LoRA, proprio projector, action-head base
```

Paired normal-vs-cross-sample diagnostic:

```text
paired_pred_l1:         0.0
paired_pred_rmse:       0.0
paired_pred_xyz_l2:     0.0
paired_pred_rpy_l2:     0.0
paired_pred_gripper_abs: 0.0
```

Decision:

NO-GO. The heatmap probe is still useful evidence that projected spatial targets are depth-sensitive, but the four-value projected-UV coordinate bottleneck does not make the final action depend on real depth. Do not run rollout or scale this checkpoint.

Next gate:

Use full heatmap logits/soft-argmax features or a coarse 3D action map as the action-grounded bottleneck, then require nonzero paired normal-vs-cross action deltas before rollout.

## Full Heatmap Policy Path

Implemented the next bottleneck:

```text
aux_target: projected_keypose_heatmap
aux_output_dim: 512
aux_heatmap_size: 16
aux_heatmap_sigma: 1.5
label shape: (2, 16, 16)
```

The dataset computes two Gaussian heatmap labels by projecting `rlbench_keypose_action[:3]` into `agentview` and `eye_in_hand`. The action head reshapes the 512D spatial output into full heatmap logits, computes soft-argmax UV coordinates, and feeds both logits and UV coordinates into the bounded action residual.

Verification:

- `py_compile` passed for `vla-scripts/finetune_depthvla.py`, `prismatic/models/action_heads.py`, and smoke script.
- Runner `bash -n` passed.
- Synthetic HDF5 dataset smoke passed with heatmap aux shape `(2, 8, 8)`.
- Action-head tensor smoke passed with logits `(2, 2, 16, 16)` and residual `(2, 8, 7)`.
- Real `open_drawer` `MAX_STEPS=1` training smoke passed with aux prediction/label shape `(1, 2, 16, 16)`.

Next gate:

Train the formal `open_drawer` heatmap policy checkpoint for `5000` steps from the RGB-only anchor. Reject before rollout unless paired normal-vs-cross-sample action deltas become nonzero.
