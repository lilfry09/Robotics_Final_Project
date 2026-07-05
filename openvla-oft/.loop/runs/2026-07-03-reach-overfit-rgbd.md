# Experiment Report: Reach-Only RGB-D Dense/Keypose Overfit

## Setup

- Date: 2026-07-03
- Dataset: `/root/RLBench/rgbd_hdf5_reach_3demos_64`
- Task: `reach_target`
- Demos/transitions: `3 / 120`
- Model: OpenVLA-OFT RGB-D dense point + object query + absolute keypose auxiliary
- Run root: `/root/runs_rlbench_reach_3demos`

## Command

```bash
HDF5_DIR=/root/RLBench/rgbd_hdf5_reach_3demos_64 \
DATASET_NAME=rlbench_reach_3demos_64 \
RUN_ROOT_DIR=/root/runs_rlbench_reach_3demos \
MAX_STEPS=5000 \
SAVE_FREQ=1000 \
DEPTH_AUX_SPATIAL_LOSS_WEIGHT=0.2 \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgbd
```

## Result

- Status: completed.
- First-step sanity:
  - depth token/context shape: `(1, 2048, 4096)`
  - action fusion gate: `1`
  - main action L1 loss: `0.699219`
  - auxiliary target: `absolute_keypose`
  - auxiliary spatial loss: `0.257812`
  - auxiliary loss weight: `0.2`
- Step `1000/2000/3000/4000/5000` checkpoints saved successfully.

Offline action diagnostic on `reach_target`, `60` samples:

| depth mode | xyz RMSE | xyz direction cosine | gripper abs error |
|---|---:|---:|---:|
| normal | `0.001901` | `0.939111` | `0.063932` |
| null | `0.001791` | `0.849990` | `0.069922` |
| shuffle | `0.001901` | `0.939143` | `0.063867` |

Closed-loop eval on stored `reach_target` episode, horizon `150`:

| depth mode | MAX_DELTA_XYZ | success | length |
|---|---:|---:|---:|
| normal | `0.03` | `0/1` | `150` |
| normal | `0.05` | `0/1` | `150` |
| normal | `0.08` | `0/1` | `150` |
| null | `0.05` | `1/1` | `31` |
| shuffle | `0.05` | `0/1` | `150` |

## Interpretation

This run is a `NO-GO` for the current RGB-D fusion recipe.

Key observations:

- RGB-only overfit succeeds on the same task, so the RLBench action adapter is not fundamentally broken.
- RGB-D normal fails at all tested `MAX_DELTA_XYZ` values, while RGB-D null succeeds at `0.05`.
- Normal and shuffle are nearly identical in offline action diagnostic, which confirms pixel shuffle is too weak for dense point tokens.
- The normal rollout failure has no `InvalidActionError`; it runs for the full horizon with small deltas, so this is likely closed-loop direction/goal bias rather than clipping.

## Next Step

- Do not scale to stable6 yet.
- Change the RGB-D fusion so normal depth cannot pull the policy away from the successful RGB anchor.
- Add a stronger cross-sample depth corruption mode for rollout/diagnostic, because pixel shuffle preserves too much dense-point structure.
