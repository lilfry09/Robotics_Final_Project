#!/usr/bin/env bash
# Reproducible DepthVLA causality entrypoint for the current codebase.
#
# Usage:
#   ./run_depth_fix_experiment.sh smoke
#   ./run_depth_fix_experiment.sh dry-run
#   ./run_depth_fix_experiment.sh train
#   ./run_depth_fix_experiment.sh eval /path/to/checkpoint
#   ./run_depth_fix_experiment.sh all
#
# The script intentionally calls the real current training entrypoint
# vla-scripts/finetune_depthvla.py. It does not depend on the removed
# finetune_depthvla_advanced.py prototype.

set -euo pipefail

MODE="${1:-smoke}"
if [[ "$MODE" == "eval" ]]; then
  EVAL_CHECKPOINT="${2:-}"
else
  EVAL_CHECKPOINT="${2:-}"
fi

BASE_DIR="${BASE_DIR:-/root/autodl-tmp/openvla-oft}"
PYTHON="${PYTHON:-/root/miniconda3/envs/depthvla/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/envs/depthvla/bin/torchrun}"

BASE_MODEL="${BASE_MODEL:-openvla/openvla-7b}"
HF_OPENVLA_SNAPSHOT_ROOT="${HF_OPENVLA_SNAPSHOT_ROOT:-${HF_HOME:-/root/autodl-tmp/hf-cache}/hub/models--openvla--openvla-7b/snapshots}"
RGB_CHECKPOINT="${RGB_CHECKPOINT:-${BASE_DIR}/runs_depthvla_stage2/openvla-7b+libero_spatial_rgbd_5tasks_20demos+rgb-only+b1+lr-0.0001+lora-r4+dropout-0.0}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/LIBERO/libero/datasets/libero_spatial_rgbd_5tasks_20demos}"
DATASET_NAME="${DATASET_NAME:-libero_spatial_rgbd_5tasks_20demos}"

RUN_ROOT="${RUN_ROOT:-${BASE_DIR}/runs_depth_fix}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-depth-fix-action-summary-${TIMESTAMP}}"
CHECKPOINT="${RUN_ROOT}/${RUN_ID}"
LOG_ROOT="${LOG_ROOT:-${BASE_DIR}/experiments/depth_fix_${TIMESTAMP}}"

MAX_STEPS="${MAX_STEPS:-3000}"
SMOKE_STEPS="${SMOKE_STEPS:-1}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-4}"

DEPTH_INTEGRATION_MODE="${DEPTH_INTEGRATION_MODE:-depth_action_summary_aux}"
DEPTH_GRID_SIZE="${DEPTH_GRID_SIZE:-4}"
DEPTH_HIDDEN_DIM="${DEPTH_HIDDEN_DIM:-256}"
DEPTH_GATE_INIT="${DEPTH_GATE_INIT:-1.0}"
DEPTH_AUX_WEIGHT="${DEPTH_AUX_WEIGHT:-0.05}"
DEPTH_AUX_TARGET="${DEPTH_AUX_TARGET:-gripper_to_contact_distance}"
DEPTH_AUX_OUTPUT_DIM="${DEPTH_AUX_OUTPUT_DIM:-1}"

TASK_IDS="${TASK_IDS:-0,1,2,7,9}"
NUM_TRIALS="${NUM_TRIALS:-3}"
ABLATION_MODES="${ABLATION_MODES:-none null shuffle_tokens shuffle_geometry}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  sed -n '2,11p' "$0"
  cat <<EOF

Modes:
  smoke          Run a 1-step training save/load smoke by default.
  dry-run        Check paths and print the train command without running it.
  train          Train the current depth-action-summary path.
  eval CHECKPOINT
                 Run none/null/shuffle_tokens ablations on CHECKPOINT.
  all            Train, then evaluate the produced latest checkpoint.

Common overrides:
  MAX_STEPS=5000 BATCH_SIZE=1 NUM_TRIALS=1 ./run_depth_fix_experiment.sh all
  DEPTH_AUX_TARGET=none DEPTH_AUX_WEIGHT=0 ./run_depth_fix_experiment.sh smoke
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "$path" ]] || die "${label} not found: ${path}"
}

supports_config_field() {
  local field="$1"
  rg -q "^[[:space:]]*${field}:" "${BASE_DIR}/vla-scripts/finetune_depthvla.py"
}

resolve_base_model_checkpoint() {
  if [[ "$BASE_MODEL" == "openvla/openvla-7b" && -d "$HF_OPENVLA_SNAPSHOT_ROOT" ]]; then
    local snapshot
    snapshot="$(find "$HF_OPENVLA_SNAPSHOT_ROOT" -maxdepth 1 -mindepth 1 -type d | sort | tail -1)"
    if [[ -n "$snapshot" ]]; then
      echo "$snapshot"
      return
    fi
  fi
  echo "$BASE_MODEL"
}

print_header() {
  echo
  echo "============================================================"
  echo "$*"
  echo "============================================================"
}

preflight() {
  require_path "$BASE_DIR" "BASE_DIR"
  require_path "$PYTHON" "PYTHON"
  require_path "$TORCHRUN" "TORCHRUN"
  require_path "$DATA_DIR" "DATA_DIR"
  require_path "$RGB_CHECKPOINT" "RGB_CHECKPOINT"
  require_path "${BASE_DIR}/vla-scripts/finetune_depthvla.py" "training script"
  require_path "${BASE_DIR}/experiments/robot/libero/run_libero_eval.py" "eval script"
  mkdir -p "$RUN_ROOT" "$LOG_ROOT"
}

build_train_args() {
  local steps="$1"
  TRAIN_ARGS=(
    --vla_path "$BASE_MODEL"
    --rgbd_data_dir "$DATA_DIR"
    --dataset_name "$DATASET_NAME"
    --run_root_dir "$RUN_ROOT"
    --run_id_override "$RUN_ID"
    --resume_components_from "$RGB_CHECKPOINT"
    --depth_integration_mode "$DEPTH_INTEGRATION_MODE"
    --use_depth True
    --depth_grid_size "$DEPTH_GRID_SIZE"
    --depth_hidden_dim "$DEPTH_HIDDEN_DIM"
    --depth_action_fusion_gate_init "$DEPTH_GATE_INIT"
    --depth_aux_spatial_loss_weight "$DEPTH_AUX_WEIGHT"
    --aux_target "$DEPTH_AUX_TARGET"
    --aux_output_dim "$DEPTH_AUX_OUTPUT_DIM"
    --freeze_vla_lora True
    --freeze_proprio_projector True
    --freeze_action_head_base True
    --batch_size "$BATCH_SIZE"
    --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS"
    --max_steps "$steps"
    --save_freq "$SAVE_FREQ"
    --save_latest_checkpoint_only True
    --merge_lora_during_training False
    --lora_rank "$LORA_RANK"
    --learning_rate "$LEARNING_RATE"
    --use_wandb False
  )

  if supports_config_field depth_dropout; then
    TRAIN_ARGS+=(--depth_dropout "${DEPTH_DROPOUT:-0.3}")
  else
    echo "Note: current finetune_depthvla.py has no depth_dropout config field; not passing dropout args."
  fi

  if supports_config_field depth_alpha_init && [[ -n "${DEPTH_ALPHA_INIT:-}" ]]; then
    TRAIN_ARGS+=(--depth_alpha_init "$DEPTH_ALPHA_INIT")
  fi

  if supports_config_field freeze_depth_alpha; then
    TRAIN_ARGS+=(--freeze_depth_alpha "${FREEZE_DEPTH_ALPHA:-False}")
  fi

  if supports_config_field min_depth_alpha && [[ -n "${MIN_DEPTH_ALPHA:-}" ]]; then
    TRAIN_ARGS+=(--min_depth_alpha "$MIN_DEPTH_ALPHA")
  fi

  if supports_config_field use_contrastive; then
    TRAIN_ARGS+=(
      --use_contrastive "${USE_CONTRASTIVE:-True}"
      --contrastive_weight "${CONTRASTIVE_WEIGHT:-0.3}"
      --contrastive_margin "${CONTRASTIVE_MARGIN:-0.05}"
    )
  else
    echo "Note: current finetune_depthvla.py has no contrastive config fields; not passing contrastive args."
  fi

  if supports_config_field null_to_base_weight; then
    TRAIN_ARGS+=(--null_to_base_weight "${NULL_TO_BASE_WEIGHT:-0.0}")
  fi

  if supports_config_field corrupt_to_base_weight; then
    TRAIN_ARGS+=(
      --corrupt_to_base_weight "${CORRUPT_TO_BASE_WEIGHT:-0.0}"
      --corrupt_depth_mode "${CORRUPT_DEPTH_MODE:-shuffle_geometry}"
    )
  fi
}

run_train() {
  local steps="$1"
  preflight
  build_train_args "$steps"

  print_header "Training DepthVLA depth-fix run"
  echo "Mode: ${MODE}"
  echo "Run ID: ${RUN_ID}"
  echo "Checkpoint: ${CHECKPOINT}"
  echo "Dataset: ${DATA_DIR}"
  echo "Resume RGB components: ${RGB_CHECKPOINT}"
  echo "Aux target: ${DEPTH_AUX_TARGET} (dim=${DEPTH_AUX_OUTPUT_DIM}, weight=${DEPTH_AUX_WEIGHT})"
  echo "Depth alpha: init=${DEPTH_ALPHA_INIT:-default}, freeze=${FREEZE_DEPTH_ALPHA:-False}, min=${MIN_DEPTH_ALPHA:-none}"
  echo "Bad-depth regularization: null_to_base=${NULL_TO_BASE_WEIGHT:-0.0}, corrupt_to_base=${CORRUPT_TO_BASE_WEIGHT:-0.0}, corrupt_mode=${CORRUPT_DEPTH_MODE:-shuffle_geometry}"
  echo "Steps: ${steps}"

  cd "$BASE_DIR"
  "$TORCHRUN" --standalone --nnodes 1 --nproc_per_node 1 \
    vla-scripts/finetune_depthvla.py "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LOG_ROOT}/train_${steps}.log"

  require_path "${CHECKPOINT}/action_head--latest_checkpoint.pt" "trained action head checkpoint"
  require_path "${CHECKPOINT}/lora_adapter" "trained LoRA adapter"
  echo "Training completed: ${CHECKPOINT}"
}

run_dry_run() {
  preflight
  build_train_args "$MAX_STEPS"

  print_header "Dry run"
  echo "Run ID: ${RUN_ID}"
  echo "Checkpoint: ${CHECKPOINT}"
  echo "Log root: ${LOG_ROOT}"
  echo "Training command:"
  printf '  %q' "$TORCHRUN" --standalone --nnodes 1 --nproc_per_node 1 vla-scripts/finetune_depthvla.py "${TRAIN_ARGS[@]}"
  echo
}

run_eval() {
  local checkpoint="$1"
  local eval_base_model
  [[ -n "$checkpoint" ]] || die "eval mode requires a checkpoint path"
  require_path "$checkpoint" "checkpoint"
  require_path "${checkpoint}/action_head--latest_checkpoint.pt" "action head checkpoint"
  preflight
  eval_base_model="$(resolve_base_model_checkpoint)"

  print_header "Evaluating DepthVLA ablations"
  echo "Checkpoint: ${checkpoint}"
  echo "Base model checkpoint: ${eval_base_model}"
  echo "Task IDs: ${TASK_IDS}"
  echo "Trials per task: ${NUM_TRIALS}"
  echo "Ablation modes: ${ABLATION_MODES}"
  if [[ "$eval_base_model" == "$BASE_MODEL" && "$BASE_MODEL" == "openvla/openvla-7b" ]]; then
    echo "Note: no local OpenVLA snapshot found under ${HF_OPENVLA_SNAPSHOT_ROOT}; eval will use the HF model id."
  fi

  cd "$BASE_DIR"
  for ablation in ${ABLATION_MODES}; do
    echo
    echo "--- Eval ablation: ${ablation} ---"
    "$PYTHON" experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "$checkpoint" \
      --base_model_checkpoint "$eval_base_model" \
      --processor_checkpoint "$eval_base_model" \
      --task_suite_name libero_spatial \
      --task_ids "$TASK_IDS" \
      --num_trials_per_task "$NUM_TRIALS" \
      --depth_integration_mode "$DEPTH_INTEGRATION_MODE" \
      --use_depth True \
      --depth_grid_size "$DEPTH_GRID_SIZE" \
      --depth_hidden_dim "$DEPTH_HIDDEN_DIM" \
      --depth_action_fusion_gate_init "$DEPTH_GATE_INIT" \
      --aux_output_dim "$DEPTH_AUX_OUTPUT_DIM" \
      --lora_rank "$LORA_RANK" \
      --unnorm_key "$DATASET_NAME" \
      --depth_ablation_mode "$ablation" \
      --run_id_note "depth-fix-${RUN_ID}-${ablation}" \
      --use_wandb False \
      2>&1 | tee "${LOG_ROOT}/eval_${ablation}.log"
  done

  print_header "Ablation summary"
  for log in "${LOG_ROOT}"/eval_*.log; do
    echo "== ${log} =="
    rg "overall success rate|Total episodes|Total successes" "$log" || true
  done
}

case "$MODE" in
  smoke)
    SAVE_FREQ="$SMOKE_STEPS"
    run_train "$SMOKE_STEPS"
    ;;
  dry-run)
    run_dry_run
    ;;
  train)
    run_train "$MAX_STEPS"
    ;;
  eval)
    run_eval "$EVAL_CHECKPOINT"
    ;;
  all)
    run_train "$MAX_STEPS"
    run_eval "$CHECKPOINT"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    die "unknown mode: ${MODE}"
    ;;
esac
