#!/usr/bin/env bash
set -euo pipefail

# Start a lightweight Xvfb display for headless RLBench/CoppeliaSim runs.
#
# Usage:
#   source experiments/robot/rlbench/source_rlbench_env.sh
#   experiments/robot/rlbench/ensure_xvfb.sh

DISPLAY_ID="${DISPLAY_ID:-1}"
DISPLAY="${DISPLAY:-:${DISPLAY_ID}.0}"
GEOMETRY="${XVFB_GEOMETRY:-1280x1024x24}"
LOG_FILE="${XVFB_LOG_FILE:-/tmp/depthvla_xvfb_${DISPLAY_ID}.log}"

if command -v xdpyinfo >/dev/null 2>&1 && xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
  echo "[ok] X display is already available: ${DISPLAY}"
  exit 0
fi

if ! command -v Xvfb >/dev/null 2>&1; then
  echo "[error] Xvfb is not installed. Install package: xvfb" >&2
  exit 2
fi

echo "[start] Xvfb ${DISPLAY} ${GEOMETRY}"
nohup Xvfb "${DISPLAY}" -screen 0 "${GEOMETRY}" -ac +extension GLX +render -noreset >"${LOG_FILE}" 2>&1 &
sleep 2

if command -v xdpyinfo >/dev/null 2>&1 && xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
  echo "[ok] X display is available: ${DISPLAY}"
  echo "[log] ${LOG_FILE}"
  exit 0
fi

echo "[error] Xvfb did not become available. See ${LOG_FILE}" >&2
exit 3
