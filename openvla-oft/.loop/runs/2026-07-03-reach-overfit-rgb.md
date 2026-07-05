# Experiment Report: Reach-Only RGB Baseline Overfit

## Setup

- Date: 2026-07-03
- Dataset: `/root/RLBench/rgbd_hdf5_reach_3demos_64`
- Task: `reach_target`
- Demos/transitions: `3 / 120`
- Model: OpenVLA-OFT RGB-only, LoRA rank 4
- Run root: `/root/runs_rlbench_reach_3demos`

## Command

```bash
HDF5_DIR=/root/RLBench/rgbd_hdf5_reach_3demos_64 \
DATASET_NAME=rlbench_reach_3demos_64 \
RUN_ROOT_DIR=/root/runs_rlbench_reach_3demos \
MAX_STEPS=5000 \
SAVE_FREQ=1000 \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgb
```

## Result

- Status: completed.
- Step 1000 checkpoint saved successfully.
- Step 2000 checkpoint saved successfully.
- Final latest checkpoint saved successfully at `2026-07-03 22:35:04`.

Closed-loop eval on stored `reach_target` episode, horizon `150`:

| MAX_DELTA_XYZ | success | length | result file |
|---:|---:|---:|---|
| `0.03` | `1/1` | `29` | `experiments/logs/rlbench_eval_reach_overfit_rgb_h150_d003/rgb_only.json` |
| `0.05` | `1/1` | `29` | `experiments/logs/rlbench_eval_reach_overfit_rgb_h150_d005/rgb_only.json` |
| `0.08` | `1/1` | `29` | `experiments/logs/rlbench_eval_reach_overfit_rgb_h150_d008/rgb_only.json` |

## Interpretation

This run passes the action-adapter sanity gate. OpenVLA-OFT RGB-only can overfit the smallest RLBench closed-loop task, so the current RLBench action adapter is not fundamentally broken.

## Next Step

- Train reach-only RGB-D dense/keypose checkpoint.
- Evaluate RGB-D normal/null/shuffle against this RGB-only baseline.
