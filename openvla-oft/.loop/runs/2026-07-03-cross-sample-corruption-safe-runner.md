# Run Report: Cross-Sample Depth Corruption and Safe RGB-D Runner

## Setup

- Date: 2026-07-03
- Scope: RLBench RGB-D eval/diagnostic runner and training runner.
- Motivation: pixel shuffle is too weak for dense point tokens, and RGB-D normal can perturb a successful RGB policy.

## Changes

- Added `cross_sample` / `shuffle_samples` / `replace_from_other_episode` depth modes to:
  - `experiments/robot/rlbench/eval_openvla_rlbench.py`
  - `experiments/robot/rlbench/diagnose_policy_actions.py`
- Added stage-runner commands:
  - `eval-rgbd-cross-sample`
  - `eval-rgbd-all-strict`
  - `diagnose-rgbd-cross-sample`
  - `diagnose-rgbd-all-strict`
- Exposed safe RGB-D training controls in `run_rlbench_rgbd_stage.sh`:
  - `DEPTH_ACTION_FUSION_GATE_INIT`
  - `DEPTH_FUSION_GATE_OVERRIDE`
  - `DEPTH_DROPOUT`
  - `FREEZE_VLA_LORA`
  - `FREEZE_PROPRIO_PROJECTOR`
  - `FREEZE_ACTION_HEAD_BASE`

## Verification

- `bash -n experiments/robot/rlbench/run_rlbench_rgbd_stage.sh`: passed.
- `python -m py_compile experiments/robot/rlbench/eval_openvla_rlbench.py experiments/robot/rlbench/diagnose_policy_actions.py`: passed.
- Offline cross-sample smoke:

```bash
DIAG_MAX_SAMPLES=4 \
DIAG_MAX_SAMPLES_PER_TASK=4 \
DEPTH_CORRUPT_BANK_SIZE=16 \
DEPTH_CORRUPT_BANK_STRIDE=5 \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh diagnose-rgbd-cross-sample
```

Result:

- Collected `16` cross-sample depth entries.
- Wrote `experiments/logs/rlbench_reach_rgbd_diag_cross_sample_smoke/rlbench_policy_action_diag_rgbd_cross_sample.json`.

Rollout cross-sample smoke:

```bash
EVAL_EPISODE_LENGTH=5 \
DEPTH_CORRUPT_EPISODE_OFFSET=1 \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh eval-rgbd-cross-sample
```

Result:

- Completed without error.
- Wrote `experiments/logs/rlbench_eval_cross_sample_smoke/rgbd_cross_sample.json`.

## Next Step

Run a safe reach-only RGB-D repair experiment:

```bash
HDF5_DIR=/root/RLBench/rgbd_hdf5_reach_3demos_64 \
DATASET_NAME=rlbench_reach_3demos_64 \
RUN_ROOT_DIR=/root/runs_rlbench_reach_safe \
MAX_STEPS=5000 \
SAVE_FREQ=1000 \
DEPTH_ACTION_FUSION_GATE_INIT=0.01 \
DEPTH_AUX_SPATIAL_LOSS_WEIGHT=0.2 \
DEPTH_DROPOUT=0.2 \
FREEZE_VLA_LORA=True \
FREEZE_PROPRIO_PROJECTOR=True \
FREEZE_ACTION_HEAD_BASE=True \
experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgbd
```

Success gate:

- RGB-D normal reaches `1/1` on `reach_target`.
- RGB-D normal does not underperform RGB-D null.
- `cross_sample` is clearly worse than normal in diagnostic or rollout.
