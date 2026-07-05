#!/usr/bin/env bash
set -euo pipefail

# Install ManiSkill3 without modifying the existing depthvla environment.
#
# Usage:
#   DRY_RUN=1 experiments/robot/maniskill/setup_maniskill_env.sh
#   experiments/robot/maniskill/setup_maniskill_env.sh
#
# The official ManiSkill docs recommend:
#   pip install --upgrade mani_skill torch
#
# We default to a separate venv because the current depthvla env is already
# tuned for OpenVLA/RLBench experiments.

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-/root/autodl-tmp/envs/maniskill3-venv}"
MS_ASSET_DIR="${MS_ASSET_DIR:-/root/autodl-tmp/maniskill_data}"
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  echo
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

mkdir -p "$(dirname "${VENV_DIR}")" "${MS_ASSET_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  run_cmd "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

run_cmd "${VENV_DIR}/bin/python" -m pip install --upgrade pip
run_cmd "${VENV_DIR}/bin/python" -m pip install --upgrade mani_skill torch h5py

echo
echo "Add these exports before running ManiSkill data collection:"
echo "export MS_ASSET_DIR=${MS_ASSET_DIR}"
echo "export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1"
echo "export PYTHON_BIN=${VENV_DIR}/bin/python"
echo
run_cmd "${VENV_DIR}/bin/python" experiments/robot/maniskill/check_maniskill_env.py
