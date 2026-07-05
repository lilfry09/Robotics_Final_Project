#!/usr/bin/env bash
set -euo pipefail

# RLBench RGB-D stage runner for the next DepthVLA-OFT experiment.
#
# Usage:
#   DRY_RUN=1 ./experiments/robot/rlbench/run_rlbench_rgbd_stage.sh all-gates
#   ./experiments/robot/rlbench/run_rlbench_rgbd_stage.sh train-rgbd
#
# Required before real execution:
#   - RLBench/PyRep/peract helpers installed
#   - CoppeliaSim environment variables exported
#   - RLBench demonstrations available under DATA_ROOT

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/depthvla/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/depthvla/bin/torchrun}"

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/openvla-oft}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/RLBench/peract_dataset/all_variations_128}"
HDF5_DIR="${HDF5_DIR:-/root/autodl-tmp/RLBench/rgbd_hdf5_6tasks_10demos}"
SUBSET_HDF5_SOURCE="${SUBSET_HDF5_SOURCE:-${HDF5_DIR}}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/root/autodl-tmp/openvla-oft/runs_rlbench_rgbd}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/openvla-oft/experiments/logs}"
EVAL_RESULT_DIR="${EVAL_RESULT_DIR:-${LOG_DIR}/rlbench_eval_results}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-${DATA_ROOT}}"
RUN_ID_NOTE="${RUN_ID_NOTE:-rlbench-rgbd-dense-keypose}"
EVAL_TRACE_OUTPUT="${EVAL_TRACE_OUTPUT:-}"
EVAL_TRACE_MAX_STEPS="${EVAL_TRACE_MAX_STEPS:-0}"

VLA_PATH="${VLA_PATH:-/root/autodl-tmp/hf-cache/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f}"
BASE_MODEL_CHECKPOINT="${BASE_MODEL_CHECKPOINT:-${VLA_PATH}}"
RGB_CHECKPOINT="${RGB_CHECKPOINT:-}"
RGBD_CHECKPOINT="${RGBD_CHECKPOINT:-}"
DATASET_NAME="${DATASET_NAME:-rlbench_rgbd_6tasks_10demos}"
TASKS="${TASKS:-slide_block_to_target,turn_tap,close_jar,open_drawer,reach_target,pick_up_cup}"
SUBSET_TASKS="${SUBSET_TASKS:-${TASKS}}"
CAMERAS="${CAMERAS:-front,wrist}"
MAX_DEMOS_PER_TASK="${MAX_DEMOS_PER_TASK:-10}"
IMAGE_SIZE="${IMAGE_SIZE:-128,128}"
RL_BENCH_DATASET_GENERATOR="${RL_BENCH_DATASET_GENERATOR:-/root/autodl-tmp/BridgeVLA/finetune/bridgevla/libs/RLBench/tools/dataset_generator.py}"

MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
EVAL_EPISODE_LENGTH="${EVAL_EPISODE_LENGTH:-150}"
EVAL_IMAGE_SIZE="${EVAL_IMAGE_SIZE:-64}"
DEPTH_POINTS_PER_VIEW="${DEPTH_POINTS_PER_VIEW:-1024}"
PROBE_POINTS_PER_VIEW="${PROBE_POINTS_PER_VIEW:-512}"
HEATMAP_PROBE_MAP_SIZE="${HEATMAP_PROBE_MAP_SIZE:-64}"
HEATMAP_PROBE_SIGMA="${HEATMAP_PROBE_SIGMA:-2.5}"
HEATMAP_PROBE_MAX_SAMPLES="${HEATMAP_PROBE_MAX_SAMPLES:-}"
HEATMAP_PROBE_STRIDE="${HEATMAP_PROBE_STRIDE:-1}"
HEATMAP_PROBE_EPOCHS="${HEATMAP_PROBE_EPOCHS:-100}"
HEATMAP_PROBE_BATCH_SIZE="${HEATMAP_PROBE_BATCH_SIZE:-32}"
HEATMAP_PROBE_LR="${HEATMAP_PROBE_LR:-3e-4}"
HEATMAP_PROBE_OUTPUT="${HEATMAP_PROBE_OUTPUT:-${LOG_DIR}/rlbench_projected_keypose_heatmap_probe.json}"
DEPTH_ACTION_FUSION_GATE_INIT="${DEPTH_ACTION_FUSION_GATE_INIT:-1.0}"
DEPTH_FUSION_GATE_OVERRIDE="${DEPTH_FUSION_GATE_OVERRIDE:-}"
DEPTH_HIDDEN_DELTA_CLIP="${DEPTH_HIDDEN_DELTA_CLIP:-0.0}"
DEPTH_ACTION_RESIDUAL_CLIP="${DEPTH_ACTION_RESIDUAL_CLIP:-0.0}"
DEPTH_KEYPOSE_RESIDUAL_WEIGHT="${DEPTH_KEYPOSE_RESIDUAL_WEIGHT:-0.0}"
DEPTH_KEYPOSE_RESIDUAL_CLIP="${DEPTH_KEYPOSE_RESIDUAL_CLIP:-0.0}"
DEPTH_POINT_ACTION_WEIGHT="${DEPTH_POINT_ACTION_WEIGHT:-0.0}"
DEPTH_POINT_ACTION_CLIP="${DEPTH_POINT_ACTION_CLIP:-0.0}"
DEPTH_WAYPOINT_ACTION_WEIGHT="${DEPTH_WAYPOINT_ACTION_WEIGHT:-0.0}"
DEPTH_WAYPOINT_ACTION_CLIP="${DEPTH_WAYPOINT_ACTION_CLIP:-0.0}"
DEPTH_WAYPOINT_ACTION_SCALE="${DEPTH_WAYPOINT_ACTION_SCALE:-1.0}"
DEPTH_WAYPOINT_ACTION_CHUNK_LEN="${DEPTH_WAYPOINT_ACTION_CHUNK_LEN:-1}"
DEPTH_AUX_SPATIAL_LOSS_WEIGHT="${DEPTH_AUX_SPATIAL_LOSS_WEIGHT:-0.05}"
DEPTH_AUX_TARGET="${DEPTH_AUX_TARGET:-absolute_keypose}"
DEPTH_AUX_OUTPUT_DIM="${DEPTH_AUX_OUTPUT_DIM:-8}"
DEPTH_AUX_FUTURE_HORIZON="${DEPTH_AUX_FUTURE_HORIZON:-10}"
DEPTH_AUX_HEATMAP_SIZE="${DEPTH_AUX_HEATMAP_SIZE:-16}"
DEPTH_AUX_HEATMAP_SIGMA="${DEPTH_AUX_HEATMAP_SIGMA:-1.5}"
DEPTH_DROPOUT="${DEPTH_DROPOUT:-0.0}"
FREEZE_VLA_LORA="${FREEZE_VLA_LORA:-False}"
FREEZE_PROPRIO_PROJECTOR="${FREEZE_PROPRIO_PROJECTOR:-False}"
FREEZE_ACTION_HEAD_BASE="${FREEZE_ACTION_HEAD_BASE:-False}"
RESUME_COMPONENTS_FROM="${RESUME_COMPONENTS_FROM:-}"
RESUME_STEP="${RESUME_STEP:-}"
DEPTH_CORRUPT_EPISODE_OFFSET="${DEPTH_CORRUPT_EPISODE_OFFSET:-1}"
DEPTH_CORRUPT_BANK_SIZE="${DEPTH_CORRUPT_BANK_SIZE:-2048}"
DEPTH_CORRUPT_BANK_STRIDE="${DEPTH_CORRUPT_BANK_STRIDE:-10}"
MAX_DELTA_XYZ="${MAX_DELTA_XYZ:-0.08}"
MAX_DELTA_RPY="${MAX_DELTA_RPY:-}"
ACTION_CHUNK_EXEC_HORIZON="${ACTION_CHUNK_EXEC_HORIZON:-1}"
GRIPPER_OVERRIDE_MODE="${GRIPPER_OVERRIDE_MODE:-none}"
GRIPPER_CLOSE_AFTER_STEP="${GRIPPER_CLOSE_AFTER_STEP:-75}"
GRIPPER_CLOSE_DISTANCE="${GRIPPER_CLOSE_DISTANCE:-0.03}"
DEPTH_POINT_LATCH_MODE="${DEPTH_POINT_LATCH_MODE:-none}"
LATCHED_DEPTH_POINT_ACTION_STEP="${LATCHED_DEPTH_POINT_ACTION_STEP:-0.0}"
LATCHED_DEPTH_POINT_ZERO_RPY="${LATCHED_DEPTH_POINT_ZERO_RPY:-True}"
POST_CLOSE_PULL_DELTA_XYZ="${POST_CLOSE_PULL_DELTA_XYZ:-}"
POST_CLOSE_PULL_STEPS="${POST_CLOSE_PULL_STEPS:-0}"
POST_CLOSE_PULL_DELAY_STEPS="${POST_CLOSE_PULL_DELAY_STEPS:-0}"
POST_CLOSE_PULL_ZERO_RPY="${POST_CLOSE_PULL_ZERO_RPY:-True}"
POST_CLOSE_DEMO_TAIL_MODE="${POST_CLOSE_DEMO_TAIL_MODE:-none}"
POST_CLOSE_DEMO_TAIL_PRECLOSE_STEPS="${POST_CLOSE_DEMO_TAIL_PRECLOSE_STEPS:-0}"
POST_CLOSE_DEMO_TAIL_STRIDE="${POST_CLOSE_DEMO_TAIL_STRIDE:-1}"
DIAG_MAX_SAMPLES="${DIAG_MAX_SAMPLES:-128}"
DIAG_MAX_SAMPLES_PER_TASK="${DIAG_MAX_SAMPLES_PER_TASK:-8}"
DIAG_STRIDE="${DIAG_STRIDE:-20}"
DIAG_COMPARE_DEPTH_MODE="${DIAG_COMPARE_DEPTH_MODE:-}"
MIN_RGB_GAIN="${MIN_RGB_GAIN:-0.05}"
MIN_ABLATION_GAIN="${MIN_ABLATION_GAIN:-0.05}"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_ROOT}"
source experiments/robot/rlbench/source_rlbench_env.sh
mkdir -p "${LOG_DIR}" "${RUN_ROOT_DIR}"
mkdir -p "${EVAL_RESULT_DIR}"

run_cmd() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

env_check() {
  run_cmd "${PYTHON_BIN}" experiments/robot/rlbench/check_rlbench_env.py
}

xvfb_check() {
  run_cmd experiments/robot/rlbench/ensure_xvfb.sh
}

generate_demos() {
  xvfb_check
  IFS=',' read -r -a task_array <<< "${TASKS}"
  for task in "${task_array[@]}"; do
    task="$(echo "${task}" | xargs)"
    if [[ -z "${task}" ]]; then
      continue
    fi
    echo
    echo "[generate] task=${task}"
    run_cmd "${PYTHON_BIN}" "${RL_BENCH_DATASET_GENERATOR}" \
      --save_path "${DATA_ROOT}" \
      --tasks "${task}" \
      --image_size "${IMAGE_SIZE}" \
      --episodes_per_task "${MAX_DEMOS_PER_TASK}" \
      --all_variations \
      --renderer opengl3
  done
}

convert_data() {
  run_cmd "${PYTHON_BIN}" experiments/robot/rlbench/convert_rlbench_to_hdf5.py \
    --data_root "${DATA_ROOT}" \
    --target_dir "${HDF5_DIR}" \
    --split train \
    --tasks "${TASKS}" \
    --cameras "${CAMERAS}" \
    --max_demos_per_task "${MAX_DEMOS_PER_TASK}" \
    --overwrite
}

validate_data() {
  run_cmd "${PYTHON_BIN}" experiments/robot/rlbench/validate_rlbench_hdf5.py \
    --data_dir "${HDF5_DIR}" \
    --strict
}

make_hdf5_subset() {
  local source_dir="${SUBSET_HDF5_SOURCE}"
  local target_dir="${HDF5_DIR}"
  if [[ "${source_dir}" == "${target_dir}" ]]; then
    echo "[error] SUBSET_HDF5_SOURCE and HDF5_DIR must be different for make-subset." >&2
    exit 2
  fi
  mkdir -p "${target_dir}"
  IFS=',' read -r -a task_array <<< "${SUBSET_TASKS}"
  for task in "${task_array[@]}"; do
    task="$(echo "${task}" | xargs)"
    if [[ -z "${task}" ]]; then
      continue
    fi
    local matched=0
    for file_path in "${source_dir}"/*"${task}"*.hdf5 "${source_dir}"/*"${task}"*.h5; do
      if [[ -e "${file_path}" ]]; then
        matched=1
        run_cmd ln -sf "${file_path}" "${target_dir}/$(basename "${file_path}")"
      fi
    done
    if [[ "${matched}" == "0" ]]; then
      echo "[error] No HDF5 file matched task=${task} in ${source_dir}" >&2
      exit 2
    fi
  done
}

dataset_smoke() {
  run_cmd "${PYTHON_BIN}" experiments/robot/rlbench/smoke_rlbench_hdf5_dataset.py
}

keypose_probe() {
  run_cmd "${PYTHON_BIN}" experiments/robot/rlbench/probe_dense_depth_keypose.py \
    --data_dir "${HDF5_DIR}" \
    --max_samples 2000 \
    --num_points_per_view "${PROBE_POINTS_PER_VIEW}" \
    --token_dim 128 \
    --hidden_dim 256 \
    --batch_size 32 \
    --epochs 20 \
    --device cuda \
    --threshold 0.01 \
    --output "${LOG_DIR}/rlbench_dense_keypose_probe.json"
}

projected_heatmap_probe() {
  local extra_args=()
  if [[ -n "${HEATMAP_PROBE_MAX_SAMPLES}" ]]; then
    extra_args+=(--max_samples "${HEATMAP_PROBE_MAX_SAMPLES}")
  fi
  run_cmd "${PYTHON_BIN}" experiments/robot/rlbench/probe_projected_keypose_heatmap.py \
    --rgbd_data_dir "${HDF5_DIR}" \
    --map_size "${HEATMAP_PROBE_MAP_SIZE}" \
    --sigma "${HEATMAP_PROBE_SIGMA}" \
    --stride "${HEATMAP_PROBE_STRIDE}" \
    --epochs "${HEATMAP_PROBE_EPOCHS}" \
    --batch_size "${HEATMAP_PROBE_BATCH_SIZE}" \
    --learning_rate "${HEATMAP_PROBE_LR}" \
    --output_json "${HEATMAP_PROBE_OUTPUT}" \
    "${extra_args[@]}"
}

train_rgb_only() {
  run_cmd "${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc_per_node 1 \
    vla-scripts/finetune_depthvla.py \
    --vla_path "${VLA_PATH}" \
    --rgbd_data_dir "${HDF5_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --run_root_dir "${RUN_ROOT_DIR}" \
    --depth_integration_mode rgb_only \
    --batch_size "${BATCH_SIZE}" \
    --max_steps "${MAX_STEPS}" \
    --save_freq "${SAVE_FREQ}" \
    --save_latest_checkpoint_only True \
    --merge_lora_during_training False \
    --lora_rank "${LORA_RANK}" \
    --learning_rate "${LR}" \
    --use_wandb False \
    --run_id_note rlbench-rgb-only
}

train_rgbd_dense() {
  local extra_args=()
  if [[ -n "${RESUME_COMPONENTS_FROM}" ]]; then
    extra_args+=(--resume_components_from "${RESUME_COMPONENTS_FROM}")
  fi
  if [[ -n "${RESUME_STEP}" ]]; then
    extra_args+=(--resume_step "${RESUME_STEP}")
  fi
  run_cmd "${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc_per_node 1 \
    vla-scripts/finetune_depthvla.py \
    --vla_path "${VLA_PATH}" \
    --rgbd_data_dir "${HDF5_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --run_root_dir "${RUN_ROOT_DIR}" \
    --depth_integration_mode depth_object_query \
    --depth_encoder_type dense_point \
    --depth_num_points_per_view "${DEPTH_POINTS_PER_VIEW}" \
    --geometry_norm none \
    --depth_action_fusion_gate_init "${DEPTH_ACTION_FUSION_GATE_INIT}" \
    --depth_hidden_delta_clip "${DEPTH_HIDDEN_DELTA_CLIP}" \
    --depth_action_residual_clip "${DEPTH_ACTION_RESIDUAL_CLIP}" \
    --depth_keypose_residual_weight "${DEPTH_KEYPOSE_RESIDUAL_WEIGHT}" \
    --depth_keypose_residual_clip "${DEPTH_KEYPOSE_RESIDUAL_CLIP}" \
    --depth_point_action_weight "${DEPTH_POINT_ACTION_WEIGHT}" \
    --depth_point_action_clip "${DEPTH_POINT_ACTION_CLIP}" \
    --depth_waypoint_action_weight "${DEPTH_WAYPOINT_ACTION_WEIGHT}" \
    --depth_waypoint_action_clip "${DEPTH_WAYPOINT_ACTION_CLIP}" \
    --depth_waypoint_action_scale "${DEPTH_WAYPOINT_ACTION_SCALE}" \
    --depth_waypoint_action_chunk_len "${DEPTH_WAYPOINT_ACTION_CHUNK_LEN}" \
    --aux_target "${DEPTH_AUX_TARGET}" \
    --aux_output_dim "${DEPTH_AUX_OUTPUT_DIM}" \
    --aux_future_horizon "${DEPTH_AUX_FUTURE_HORIZON}" \
    --aux_heatmap_size "${DEPTH_AUX_HEATMAP_SIZE}" \
    --aux_heatmap_sigma "${DEPTH_AUX_HEATMAP_SIGMA}" \
    --depth_aux_spatial_loss_weight "${DEPTH_AUX_SPATIAL_LOSS_WEIGHT}" \
    --depth_dropout "${DEPTH_DROPOUT}" \
    --freeze_vla_lora "${FREEZE_VLA_LORA}" \
    --freeze_proprio_projector "${FREEZE_PROPRIO_PROJECTOR}" \
    --freeze_action_head_base "${FREEZE_ACTION_HEAD_BASE}" \
    --batch_size "${BATCH_SIZE}" \
    --max_steps "${MAX_STEPS}" \
    --save_freq "${SAVE_FREQ}" \
    --save_latest_checkpoint_only True \
    --merge_lora_during_training False \
    --lora_rank "${LORA_RANK}" \
    --learning_rate "${LR}" \
    --use_wandb False \
    --run_id_note "${RUN_ID_NOTE}" \
    "${extra_args[@]}"
}

require_checkpoint() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[dry-run] ${name} is unset; command will require it for real execution."
      return
    fi
    echo "[error] ${name} is unset. Point it at a trained run/checkpoint directory." >&2
    exit 2
  fi
}

eval_rgb_only() {
  local extra_args=()
  if [[ -n "${MAX_DELTA_RPY}" ]]; then
    extra_args+=(--max_delta_rpy "${MAX_DELTA_RPY}")
  fi
  if [[ "${GRIPPER_OVERRIDE_MODE}" != "none" ]]; then
    extra_args+=(
      --gripper_override_mode "${GRIPPER_OVERRIDE_MODE}"
      --gripper_close_after_step "${GRIPPER_CLOSE_AFTER_STEP}"
      --gripper_close_distance "${GRIPPER_CLOSE_DISTANCE}"
    )
  fi
  if [[ "${ACTION_CHUNK_EXEC_HORIZON}" != "1" ]]; then
    extra_args+=(--action_chunk_exec_horizon "${ACTION_CHUNK_EXEC_HORIZON}")
  fi
  if [[ "${DEPTH_POINT_LATCH_MODE}" != "none" || "${LATCHED_DEPTH_POINT_ACTION_STEP}" != "0.0" ]]; then
    extra_args+=(
      --depth_point_latch_mode "${DEPTH_POINT_LATCH_MODE}"
      --latched_depth_point_action_step "${LATCHED_DEPTH_POINT_ACTION_STEP}"
    )
    if [[ "${LATCHED_DEPTH_POINT_ZERO_RPY}" == "False" || "${LATCHED_DEPTH_POINT_ZERO_RPY}" == "false" || "${LATCHED_DEPTH_POINT_ZERO_RPY}" == "0" ]]; then
      extra_args+=(--no-latched_depth_point_zero_rpy)
    else
      extra_args+=(--latched_depth_point_zero_rpy)
    fi
  fi
  if [[ -n "${POST_CLOSE_PULL_DELTA_XYZ}" && "${POST_CLOSE_PULL_STEPS}" != "0" ]]; then
    extra_args+=(
      --post_close_pull_delta_xyz "${POST_CLOSE_PULL_DELTA_XYZ}"
      --post_close_pull_steps "${POST_CLOSE_PULL_STEPS}"
      --post_close_pull_delay_steps "${POST_CLOSE_PULL_DELAY_STEPS}"
    )
    if [[ "${POST_CLOSE_PULL_ZERO_RPY}" == "False" || "${POST_CLOSE_PULL_ZERO_RPY}" == "false" || "${POST_CLOSE_PULL_ZERO_RPY}" == "0" ]]; then
      extra_args+=(--no-post_close_pull_zero_rpy)
    else
      extra_args+=(--post_close_pull_zero_rpy)
    fi
  fi
  if [[ "${POST_CLOSE_DEMO_TAIL_MODE}" != "none" ]]; then
    extra_args+=(
      --post_close_demo_tail_mode "${POST_CLOSE_DEMO_TAIL_MODE}"
      --post_close_demo_tail_preclose_steps "${POST_CLOSE_DEMO_TAIL_PRECLOSE_STEPS}"
      --post_close_demo_tail_stride "${POST_CLOSE_DEMO_TAIL_STRIDE}"
    )
  fi
  if [[ -n "${EVAL_TRACE_OUTPUT}" ]]; then
    extra_args+=(--trace_output "${EVAL_TRACE_OUTPUT}" --trace_max_steps "${EVAL_TRACE_MAX_STEPS}")
  fi
  xvfb_check
  require_checkpoint "RGB_CHECKPOINT" "${RGB_CHECKPOINT}"
  run_cmd "${PYTHON_BIN}" experiments/robot/rlbench/eval_openvla_rlbench.py \
    --checkpoint "${RGB_CHECKPOINT}" \
    --base_model_checkpoint "${BASE_MODEL_CHECKPOINT}" \
    --eval_datafolder "${EVAL_DATA_ROOT}" \
    --tasks "${TASKS}" \
    --output "${EVAL_RESULT_DIR}/rgb_only.json" \
    --unnorm_key "${DATASET_NAME}" \
    --eval_episodes "${EVAL_EPISODES}" \
    --episode_length "${EVAL_EPISODE_LENGTH}" \
    --image_size "${EVAL_IMAGE_SIZE}" \
    --lora_rank "${LORA_RANK}" \
    --max_delta_xyz "${MAX_DELTA_XYZ}" \
    "${extra_args[@]}"
}

eval_rgbd_mode() {
  local mode="$1"
  local extra_args=()
  if [[ -n "${DEPTH_FUSION_GATE_OVERRIDE}" ]]; then
    extra_args+=(--depth_fusion_gate_override "${DEPTH_FUSION_GATE_OVERRIDE}")
  fi
  if [[ -n "${MAX_DELTA_RPY}" ]]; then
    extra_args+=(--max_delta_rpy "${MAX_DELTA_RPY}")
  fi
  if [[ "${GRIPPER_OVERRIDE_MODE}" != "none" ]]; then
    extra_args+=(
      --gripper_override_mode "${GRIPPER_OVERRIDE_MODE}"
      --gripper_close_after_step "${GRIPPER_CLOSE_AFTER_STEP}"
      --gripper_close_distance "${GRIPPER_CLOSE_DISTANCE}"
    )
  fi
  if [[ "${ACTION_CHUNK_EXEC_HORIZON}" != "1" ]]; then
    extra_args+=(--action_chunk_exec_horizon "${ACTION_CHUNK_EXEC_HORIZON}")
  fi
  if [[ "${DEPTH_POINT_LATCH_MODE}" != "none" || "${LATCHED_DEPTH_POINT_ACTION_STEP}" != "0.0" ]]; then
    extra_args+=(
      --depth_point_latch_mode "${DEPTH_POINT_LATCH_MODE}"
      --latched_depth_point_action_step "${LATCHED_DEPTH_POINT_ACTION_STEP}"
    )
    if [[ "${LATCHED_DEPTH_POINT_ZERO_RPY}" == "False" || "${LATCHED_DEPTH_POINT_ZERO_RPY}" == "false" || "${LATCHED_DEPTH_POINT_ZERO_RPY}" == "0" ]]; then
      extra_args+=(--no-latched_depth_point_zero_rpy)
    else
      extra_args+=(--latched_depth_point_zero_rpy)
    fi
  fi
  if [[ -n "${POST_CLOSE_PULL_DELTA_XYZ}" && "${POST_CLOSE_PULL_STEPS}" != "0" ]]; then
    extra_args+=(
      --post_close_pull_delta_xyz "${POST_CLOSE_PULL_DELTA_XYZ}"
      --post_close_pull_steps "${POST_CLOSE_PULL_STEPS}"
      --post_close_pull_delay_steps "${POST_CLOSE_PULL_DELAY_STEPS}"
    )
    if [[ "${POST_CLOSE_PULL_ZERO_RPY}" == "False" || "${POST_CLOSE_PULL_ZERO_RPY}" == "false" || "${POST_CLOSE_PULL_ZERO_RPY}" == "0" ]]; then
      extra_args+=(--no-post_close_pull_zero_rpy)
    else
      extra_args+=(--post_close_pull_zero_rpy)
    fi
  fi
  if [[ "${POST_CLOSE_DEMO_TAIL_MODE}" != "none" ]]; then
    extra_args+=(
      --post_close_demo_tail_mode "${POST_CLOSE_DEMO_TAIL_MODE}"
      --post_close_demo_tail_preclose_steps "${POST_CLOSE_DEMO_TAIL_PRECLOSE_STEPS}"
      --post_close_demo_tail_stride "${POST_CLOSE_DEMO_TAIL_STRIDE}"
    )
  fi
  if [[ -n "${EVAL_TRACE_OUTPUT}" ]]; then
    extra_args+=(--trace_output "${EVAL_TRACE_OUTPUT}" --trace_max_steps "${EVAL_TRACE_MAX_STEPS}")
  fi
  xvfb_check
  require_checkpoint "RGBD_CHECKPOINT" "${RGBD_CHECKPOINT}"
  run_cmd "${PYTHON_BIN}" experiments/robot/rlbench/eval_openvla_rlbench.py \
    --checkpoint "${RGBD_CHECKPOINT}" \
    --base_model_checkpoint "${BASE_MODEL_CHECKPOINT}" \
    --eval_datafolder "${EVAL_DATA_ROOT}" \
    --tasks "${TASKS}" \
    --output "${EVAL_RESULT_DIR}/rgbd_${mode}.json" \
    --unnorm_key "${DATASET_NAME}" \
    --eval_episodes "${EVAL_EPISODES}" \
    --episode_length "${EVAL_EPISODE_LENGTH}" \
    --image_size "${EVAL_IMAGE_SIZE}" \
    --lora_rank "${LORA_RANK}" \
    --use_depth \
    --depth_mode "${mode}" \
    --depth_fusion_mode object_query \
    --depth_encoder_type dense_point \
    --depth_num_points_per_view "${DEPTH_POINTS_PER_VIEW}" \
    --geometry_norm none \
    --depth_hidden_delta_clip "${DEPTH_HIDDEN_DELTA_CLIP}" \
    --depth_action_residual_clip "${DEPTH_ACTION_RESIDUAL_CLIP}" \
    --depth_keypose_residual_weight "${DEPTH_KEYPOSE_RESIDUAL_WEIGHT}" \
    --depth_keypose_residual_clip "${DEPTH_KEYPOSE_RESIDUAL_CLIP}" \
    --depth_point_action_weight "${DEPTH_POINT_ACTION_WEIGHT}" \
    --depth_point_action_clip "${DEPTH_POINT_ACTION_CLIP}" \
    --depth_waypoint_action_weight "${DEPTH_WAYPOINT_ACTION_WEIGHT}" \
    --depth_waypoint_action_clip "${DEPTH_WAYPOINT_ACTION_CLIP}" \
    --depth_waypoint_action_scale "${DEPTH_WAYPOINT_ACTION_SCALE}" \
    --depth_waypoint_action_chunk_len "${DEPTH_WAYPOINT_ACTION_CHUNK_LEN}" \
    --aux_output_dim "${DEPTH_AUX_OUTPUT_DIM}" \
    --max_delta_xyz "${MAX_DELTA_XYZ}" \
    --depth_corrupt_episode_offset "${DEPTH_CORRUPT_EPISODE_OFFSET}" \
    "${extra_args[@]}"
}

eval_rgbd_all_modes() {
  eval_rgbd_mode normal
  eval_rgbd_mode null
  eval_rgbd_mode shuffle
}

eval_rgbd_all_modes_strict() {
  eval_rgbd_mode normal
  eval_rgbd_mode null
  eval_rgbd_mode cross_sample
}

diagnose_rgb_actions() {
  require_checkpoint "RGB_CHECKPOINT" "${RGB_CHECKPOINT}"
  run_cmd "${PYTHON_BIN}" experiments/robot/rlbench/diagnose_policy_actions.py \
    --checkpoint "${RGB_CHECKPOINT}" \
    --base_model_checkpoint "${BASE_MODEL_CHECKPOINT}" \
    --data_dir "${HDF5_DIR}" \
    --tasks "${TASKS}" \
    --output "${LOG_DIR}/rlbench_policy_action_diag_rgb_only.json" \
    --unnorm_key "${DATASET_NAME}" \
    --max_samples "${DIAG_MAX_SAMPLES}" \
    --max_samples_per_task "${DIAG_MAX_SAMPLES_PER_TASK}" \
    --stride "${DIAG_STRIDE}" \
    --lora_rank "${LORA_RANK}"
}

diagnose_rgbd_actions() {
  local mode="$1"
  local extra_args=()
  if [[ -n "${DEPTH_FUSION_GATE_OVERRIDE}" ]]; then
    extra_args+=(--depth_fusion_gate_override "${DEPTH_FUSION_GATE_OVERRIDE}")
  fi
  if [[ -n "${DIAG_COMPARE_DEPTH_MODE}" ]]; then
    extra_args+=(--compare_depth_mode "${DIAG_COMPARE_DEPTH_MODE}")
  fi
  require_checkpoint "RGBD_CHECKPOINT" "${RGBD_CHECKPOINT}"
  run_cmd "${PYTHON_BIN}" experiments/robot/rlbench/diagnose_policy_actions.py \
    --checkpoint "${RGBD_CHECKPOINT}" \
    --base_model_checkpoint "${BASE_MODEL_CHECKPOINT}" \
    --data_dir "${HDF5_DIR}" \
    --tasks "${TASKS}" \
    --output "${LOG_DIR}/rlbench_policy_action_diag_rgbd_${mode}.json" \
    --unnorm_key "${DATASET_NAME}" \
    --max_samples "${DIAG_MAX_SAMPLES}" \
    --max_samples_per_task "${DIAG_MAX_SAMPLES_PER_TASK}" \
    --stride "${DIAG_STRIDE}" \
    --lora_rank "${LORA_RANK}" \
    --use_depth \
    --depth_mode "${mode}" \
    --depth_fusion_mode object_query \
    --depth_encoder_type dense_point \
    --depth_num_points_per_view "${DEPTH_POINTS_PER_VIEW}" \
    --geometry_norm none \
    --depth_hidden_delta_clip "${DEPTH_HIDDEN_DELTA_CLIP}" \
    --depth_action_residual_clip "${DEPTH_ACTION_RESIDUAL_CLIP}" \
    --depth_keypose_residual_weight "${DEPTH_KEYPOSE_RESIDUAL_WEIGHT}" \
    --depth_keypose_residual_clip "${DEPTH_KEYPOSE_RESIDUAL_CLIP}" \
    --depth_point_action_weight "${DEPTH_POINT_ACTION_WEIGHT}" \
    --depth_point_action_clip "${DEPTH_POINT_ACTION_CLIP}" \
    --depth_waypoint_action_weight "${DEPTH_WAYPOINT_ACTION_WEIGHT}" \
    --depth_waypoint_action_clip "${DEPTH_WAYPOINT_ACTION_CLIP}" \
    --depth_waypoint_action_scale "${DEPTH_WAYPOINT_ACTION_SCALE}" \
    --depth_waypoint_action_chunk_len "${DEPTH_WAYPOINT_ACTION_CHUNK_LEN}" \
    --aux_target "${DEPTH_AUX_TARGET}" \
    --aux_output_dim "${DEPTH_AUX_OUTPUT_DIM}" \
    --aux_future_horizon "${DEPTH_AUX_FUTURE_HORIZON}" \
    --depth_corrupt_bank_size "${DEPTH_CORRUPT_BANK_SIZE}" \
    --depth_corrupt_bank_stride "${DEPTH_CORRUPT_BANK_STRIDE}" \
    "${extra_args[@]}"
}

diagnose_rgbd_all_modes() {
  diagnose_rgbd_actions normal
  diagnose_rgbd_actions null
  diagnose_rgbd_actions shuffle
}

diagnose_rgbd_all_modes_strict() {
  diagnose_rgbd_actions normal
  diagnose_rgbd_actions null
  diagnose_rgbd_actions cross_sample
}

gate_results() {
  run_cmd "${PYTHON_BIN}" experiments/robot/rlbench/compare_rgbd_rollout_results.py \
    --rgb_only "${EVAL_RESULT_DIR}/rgb_only.json" \
    --rgbd_normal "${EVAL_RESULT_DIR}/rgbd_normal.json" \
    --rgbd_null "${EVAL_RESULT_DIR}/rgbd_null.json" \
    --rgbd_shuffle "${EVAL_RESULT_DIR}/rgbd_shuffle.json" \
    --min_rgb_gain "${MIN_RGB_GAIN}" \
    --min_ablation_gain "${MIN_ABLATION_GAIN}" \
    --output "${EVAL_RESULT_DIR}/rgbd_causal_gate.json"
}

print_config() {
  cat <<EOF
RLBench RGB-D stage config
  PROJECT_ROOT=${PROJECT_ROOT}
  DATA_ROOT=${DATA_ROOT}
  EVAL_DATA_ROOT=${EVAL_DATA_ROOT}
  HDF5_DIR=${HDF5_DIR}
  SUBSET_HDF5_SOURCE=${SUBSET_HDF5_SOURCE}
  RUN_ROOT_DIR=${RUN_ROOT_DIR}
  VLA_PATH=${VLA_PATH}
  BASE_MODEL_CHECKPOINT=${BASE_MODEL_CHECKPOINT}
  RGB_CHECKPOINT=${RGB_CHECKPOINT}
  RGBD_CHECKPOINT=${RGBD_CHECKPOINT}
  DATASET_NAME=${DATASET_NAME}
  TASKS=${TASKS}
  SUBSET_TASKS=${SUBSET_TASKS}
  CAMERAS=${CAMERAS}
  MAX_DEMOS_PER_TASK=${MAX_DEMOS_PER_TASK}
  IMAGE_SIZE=${IMAGE_SIZE}
  RL_BENCH_DATASET_GENERATOR=${RL_BENCH_DATASET_GENERATOR}
  MAX_STEPS=${MAX_STEPS}
  SAVE_FREQ=${SAVE_FREQ}
  BATCH_SIZE=${BATCH_SIZE}
  LR=${LR}
  LORA_RANK=${LORA_RANK}
  EVAL_EPISODES=${EVAL_EPISODES}
  EVAL_EPISODE_LENGTH=${EVAL_EPISODE_LENGTH}
  EVAL_IMAGE_SIZE=${EVAL_IMAGE_SIZE}
  DEPTH_POINTS_PER_VIEW=${DEPTH_POINTS_PER_VIEW}
  PROBE_POINTS_PER_VIEW=${PROBE_POINTS_PER_VIEW}
  HEATMAP_PROBE_MAP_SIZE=${HEATMAP_PROBE_MAP_SIZE}
  HEATMAP_PROBE_SIGMA=${HEATMAP_PROBE_SIGMA}
  HEATMAP_PROBE_MAX_SAMPLES=${HEATMAP_PROBE_MAX_SAMPLES}
  HEATMAP_PROBE_STRIDE=${HEATMAP_PROBE_STRIDE}
  HEATMAP_PROBE_EPOCHS=${HEATMAP_PROBE_EPOCHS}
  HEATMAP_PROBE_BATCH_SIZE=${HEATMAP_PROBE_BATCH_SIZE}
  HEATMAP_PROBE_LR=${HEATMAP_PROBE_LR}
  HEATMAP_PROBE_OUTPUT=${HEATMAP_PROBE_OUTPUT}
  DEPTH_ACTION_FUSION_GATE_INIT=${DEPTH_ACTION_FUSION_GATE_INIT}
  DEPTH_FUSION_GATE_OVERRIDE=${DEPTH_FUSION_GATE_OVERRIDE}
  DEPTH_HIDDEN_DELTA_CLIP=${DEPTH_HIDDEN_DELTA_CLIP}
  DEPTH_ACTION_RESIDUAL_CLIP=${DEPTH_ACTION_RESIDUAL_CLIP}
  DEPTH_KEYPOSE_RESIDUAL_WEIGHT=${DEPTH_KEYPOSE_RESIDUAL_WEIGHT}
  DEPTH_KEYPOSE_RESIDUAL_CLIP=${DEPTH_KEYPOSE_RESIDUAL_CLIP}
  DEPTH_POINT_ACTION_WEIGHT=${DEPTH_POINT_ACTION_WEIGHT}
  DEPTH_POINT_ACTION_CLIP=${DEPTH_POINT_ACTION_CLIP}
  DEPTH_WAYPOINT_ACTION_WEIGHT=${DEPTH_WAYPOINT_ACTION_WEIGHT}
  DEPTH_WAYPOINT_ACTION_CLIP=${DEPTH_WAYPOINT_ACTION_CLIP}
  DEPTH_WAYPOINT_ACTION_SCALE=${DEPTH_WAYPOINT_ACTION_SCALE}
  DEPTH_WAYPOINT_ACTION_CHUNK_LEN=${DEPTH_WAYPOINT_ACTION_CHUNK_LEN}
  DEPTH_AUX_SPATIAL_LOSS_WEIGHT=${DEPTH_AUX_SPATIAL_LOSS_WEIGHT}
  DEPTH_AUX_TARGET=${DEPTH_AUX_TARGET}
  DEPTH_AUX_OUTPUT_DIM=${DEPTH_AUX_OUTPUT_DIM}
  DEPTH_AUX_FUTURE_HORIZON=${DEPTH_AUX_FUTURE_HORIZON}
  DEPTH_AUX_HEATMAP_SIZE=${DEPTH_AUX_HEATMAP_SIZE}
  DEPTH_AUX_HEATMAP_SIGMA=${DEPTH_AUX_HEATMAP_SIGMA}
  DEPTH_DROPOUT=${DEPTH_DROPOUT}
  FREEZE_VLA_LORA=${FREEZE_VLA_LORA}
  FREEZE_PROPRIO_PROJECTOR=${FREEZE_PROPRIO_PROJECTOR}
  FREEZE_ACTION_HEAD_BASE=${FREEZE_ACTION_HEAD_BASE}
  RESUME_COMPONENTS_FROM=${RESUME_COMPONENTS_FROM}
  RESUME_STEP=${RESUME_STEP}
  DEPTH_CORRUPT_EPISODE_OFFSET=${DEPTH_CORRUPT_EPISODE_OFFSET}
  DEPTH_CORRUPT_BANK_SIZE=${DEPTH_CORRUPT_BANK_SIZE}
  DEPTH_CORRUPT_BANK_STRIDE=${DEPTH_CORRUPT_BANK_STRIDE}
  MAX_DELTA_XYZ=${MAX_DELTA_XYZ}
  MAX_DELTA_RPY=${MAX_DELTA_RPY}
  ACTION_CHUNK_EXEC_HORIZON=${ACTION_CHUNK_EXEC_HORIZON}
  EVAL_TRACE_OUTPUT=${EVAL_TRACE_OUTPUT}
  EVAL_TRACE_MAX_STEPS=${EVAL_TRACE_MAX_STEPS}
  GRIPPER_OVERRIDE_MODE=${GRIPPER_OVERRIDE_MODE}
  GRIPPER_CLOSE_AFTER_STEP=${GRIPPER_CLOSE_AFTER_STEP}
  GRIPPER_CLOSE_DISTANCE=${GRIPPER_CLOSE_DISTANCE}
  DEPTH_POINT_LATCH_MODE=${DEPTH_POINT_LATCH_MODE}
  LATCHED_DEPTH_POINT_ACTION_STEP=${LATCHED_DEPTH_POINT_ACTION_STEP}
  LATCHED_DEPTH_POINT_ZERO_RPY=${LATCHED_DEPTH_POINT_ZERO_RPY}
  POST_CLOSE_PULL_DELTA_XYZ=${POST_CLOSE_PULL_DELTA_XYZ}
  POST_CLOSE_PULL_STEPS=${POST_CLOSE_PULL_STEPS}
  POST_CLOSE_PULL_DELAY_STEPS=${POST_CLOSE_PULL_DELAY_STEPS}
  POST_CLOSE_PULL_ZERO_RPY=${POST_CLOSE_PULL_ZERO_RPY}
  POST_CLOSE_DEMO_TAIL_MODE=${POST_CLOSE_DEMO_TAIL_MODE}
  POST_CLOSE_DEMO_TAIL_PRECLOSE_STEPS=${POST_CLOSE_DEMO_TAIL_PRECLOSE_STEPS}
  POST_CLOSE_DEMO_TAIL_STRIDE=${POST_CLOSE_DEMO_TAIL_STRIDE}
  DIAG_MAX_SAMPLES=${DIAG_MAX_SAMPLES}
  DIAG_MAX_SAMPLES_PER_TASK=${DIAG_MAX_SAMPLES_PER_TASK}
  DIAG_STRIDE=${DIAG_STRIDE}
  DIAG_COMPARE_DEPTH_MODE=${DIAG_COMPARE_DEPTH_MODE}
  EVAL_RESULT_DIR=${EVAL_RESULT_DIR}
  MIN_RGB_GAIN=${MIN_RGB_GAIN}
  MIN_ABLATION_GAIN=${MIN_ABLATION_GAIN}
  DRY_RUN=${DRY_RUN}
EOF
}

usage() {
  cat <<'EOF'
Commands:
  config        Print resolved configuration.
  env-check     Check RLBench/PyRep/CoppeliaSim Python environment.
  xvfb          Start/check Xvfb display for headless RLBench.
  generate-demos Generate RLBench pilot demonstrations under DATA_ROOT.
  convert       Convert RLBench demos to DepthVLA HDF5.
  validate      Validate converted HDF5 files.
  make-subset   Symlink selected SUBSET_TASKS from SUBSET_HDF5_SOURCE into HDF5_DIR.
  smoke         Run synthetic RLBench HDF5 dataset smoke.
  probe         Run dense-depth -> absolute-keypose normal/null/shuffle probe.
  projected-heatmap-probe Run projected keypose heatmap normal/null/cross-sample probe.
  train-rgb     Train matched RGB-only baseline.
  train-rgbd    Train RGB-D dense-point + absolute-keypose model.
  eval-rgb      Evaluate RGB-only checkpoint in RLBench.
  eval-rgbd-normal   Evaluate RGB-D checkpoint with normal depth.
  eval-rgbd-null     Evaluate RGB-D checkpoint with null depth.
  eval-rgbd-shuffle  Evaluate RGB-D checkpoint with shuffled depth.
  eval-rgbd-cross-sample  Evaluate RGB-D checkpoint with depth from another episode.
  eval-rgbd-all      Evaluate RGB-D normal/null/shuffle.
  eval-rgbd-all-strict Evaluate RGB-D normal/null/cross-sample.
  diagnose-rgb       Offline policy-vs-demo action diagnostic for RGB-only.
  diagnose-rgbd-normal   Offline policy-vs-demo diagnostic with normal depth.
  diagnose-rgbd-null     Offline policy-vs-demo diagnostic with null depth.
  diagnose-rgbd-shuffle  Offline policy-vs-demo diagnostic with shuffled depth.
  diagnose-rgbd-cross-sample  Offline diagnostic with depth from other samples.
  diagnose-rgbd-all      Offline policy-vs-demo RGB-D normal/null/shuffle diagnostics.
  diagnose-rgbd-all-strict Offline policy-vs-demo RGB-D normal/null/cross-sample diagnostics.
  gate-results  Compare RGB/RGB-D normal/null/shuffle rollout result files.
  all-gates     Run env-check, convert, validate, smoke, probe.
  dry-run       Print all-gates, train-rgb, train-rgbd commands without running.
EOF
}

cmd="${1:-}"
case "${cmd}" in
  config)
    print_config
    ;;
  env-check)
    env_check
    ;;
  xvfb)
    xvfb_check
    ;;
  generate-demos)
    generate_demos
    ;;
  convert)
    convert_data
    ;;
  validate)
    validate_data
    ;;
  make-subset)
    make_hdf5_subset
    ;;
  smoke)
    dataset_smoke
    ;;
  probe)
    keypose_probe
    ;;
  projected-heatmap-probe)
    projected_heatmap_probe
    ;;
  train-rgb)
    train_rgb_only
    ;;
  train-rgbd)
    train_rgbd_dense
    ;;
  eval-rgb)
    eval_rgb_only
    ;;
  eval-rgbd-normal)
    eval_rgbd_mode normal
    ;;
  eval-rgbd-null)
    eval_rgbd_mode null
    ;;
  eval-rgbd-shuffle)
    eval_rgbd_mode shuffle
    ;;
  eval-rgbd-cross-sample)
    eval_rgbd_mode cross_sample
    ;;
  eval-rgbd-all)
    eval_rgbd_all_modes
    ;;
  eval-rgbd-all-strict)
    eval_rgbd_all_modes_strict
    ;;
  diagnose-rgb)
    diagnose_rgb_actions
    ;;
  diagnose-rgbd-normal)
    diagnose_rgbd_actions normal
    ;;
  diagnose-rgbd-null)
    diagnose_rgbd_actions null
    ;;
  diagnose-rgbd-shuffle)
    diagnose_rgbd_actions shuffle
    ;;
  diagnose-rgbd-cross-sample)
    diagnose_rgbd_actions cross_sample
    ;;
  diagnose-rgbd-all)
    diagnose_rgbd_all_modes
    ;;
  diagnose-rgbd-all-strict)
    diagnose_rgbd_all_modes_strict
    ;;
  gate-results)
    gate_results
    ;;
  all-gates)
    print_config
    env_check
    xvfb_check
    convert_data
    validate_data
    dataset_smoke
    keypose_probe
    projected_heatmap_probe
    ;;
  dry-run)
    DRY_RUN=1
    print_config
    env_check
    xvfb_check
    generate_demos
    convert_data
    validate_data
    dataset_smoke
    keypose_probe
    projected_heatmap_probe
    train_rgb_only
    train_rgbd_dense
    eval_rgb_only
    eval_rgbd_all_modes
    gate_results
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
