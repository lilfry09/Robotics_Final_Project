# 2026-07-04 Open Drawer 3D Gate

## Goal

Move the causal depth gate away from easy `reach_target` toward a task with contact and articulated 3D geometry.

## Data

- Source: `/root/RLBench/rgbd_hdf5_stable6_3demos_64`
- Subset: `/root/RLBench/rgbd_hdf5_open_drawer_3demos_64`
- Task: `open_drawer`
- Validation: `3 demos / 317 transitions`
- Keypose xyz range:
  - min `[0.0734, -0.0205, 1.0275]`
  - max `[0.3402, 0.3862, 1.4710]`

## Offline Spatial Gate

Dense depth -> absolute keypose probe with `1024` points per view:

| depth mode | keypose xyz RMSE |
|---|---:|
| normal | `0.0705` |
| null | `0.1420` |
| shuffle | `0.1466` |

Verdict: GO for matched policy training. Normal depth has clear spatial signal.

## RGB-Only Baseline

Training command:

```bash
HDF5_DIR=/root/RLBench/rgbd_hdf5_open_drawer_3demos_64 \
DATASET_NAME=rlbench_open_drawer_3demos_64 \
RUN_ROOT_DIR=/root/runs_rlbench_open_drawer_3demos \
TASKS=open_drawer \
MAX_STEPS=5000 \
SAVE_FREQ=1000 \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgb
```

Result:

- Checkpoint: `/root/runs_rlbench_open_drawer_3demos/...--rlbench-rgb-only`
- Eval: `1` episode, horizon `200`, `MAX_DELTA_XYZ=0.05`
- Success: `0/1`
- Length: `200`
- Error: none
- Mean delta xyz norm: `0.00595`

Interpretation:

RGB-only does not saturate `open_drawer`, and the failure is a timeout rather than an invalid action crash. This is a better depth-gain test than `reach_target`.

Next:

## Safe RGB-D From RGB Anchor

Training:

- Resume anchor: `/root/runs_rlbench_open_drawer_3demos/...--rlbench-rgb-only`
- Fusion: dense point tokens, object-query, gate `0.01`
- Protection: freeze VLA LoRA, proprio projector, and action-head base
- Clamps: hidden `0.05`, action residual `0.02`
- Aux: absolute keypose, weight `0.2`

Rollout, horizon `200`, `MAX_DELTA_XYZ=0.05`:

| policy / depth mode | success | length | delta xyz mean | delta rpy mean |
|---|---:|---:|---:|---:|
| RGB-only | `0/1` | `200` | `0.00595` | `0.04167` |
| RGB-D normal | `0/1` | `200` | `0.00592` | `0.04028` |
| RGB-D null | `0/1` | `200` | `0.00597` | `0.04083` |
| RGB-D cross_sample | `0/1` | `200` | `0.00608` | `0.04138` |

Offline action diagnostic:

| depth mode | xyz RMSE | xyz cosine | gripper abs error |
|---|---:|---:|---:|
| normal | `0.001336` | `0.78733` | `0.05780` |
| null | `0.001357` | `0.78683` | `0.06006` |
| cross_sample | `0.001336` | `0.78733` | `0.05780` |

Verdict: NO-GO. Open-drawer is a useful non-saturated gate, but shallow safe residual still does not make true depth geometry causally useful.

Checkpoint cleanup:

- Deleted no-go safe RGB-D checkpoint after recording results.
- Kept RGB-only anchor for future keypose/action-map methods.

Next:

1. Implement keypose-conditioned residual or 3D action-map/heatmap action grounding.
2. Require normal to beat cross-sample in offline action formation before another long rollout.
3. Only then re-run `open_drawer` closed-loop.
