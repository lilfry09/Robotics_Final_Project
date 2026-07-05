#!/usr/bin/env bash
set -euo pipefail

# Controlled RLBench setup helper.
#
# Default mode is check-only:
#   ./experiments/robot/rlbench/setup_rlbench_env.sh
#
# To clone/install missing Python packages:
#   INSTALL=1 ./experiments/robot/rlbench/setup_rlbench_env.sh
#
# To also download CoppeliaSim:
#   INSTALL=1 DOWNLOAD_COPPELIASIM=1 ./experiments/robot/rlbench/setup_rlbench_env.sh
#
# To build PyRep's CFFI extension after CoppeliaSim is present:
#   BUILD_PYREP=1 ./experiments/robot/rlbench/setup_rlbench_env.sh

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/depthvla/bin/python}"
BRIDGEVLA_FINETUNE_ROOT="${BRIDGEVLA_FINETUNE_ROOT:-/root/autodl-tmp/BridgeVLA/finetune}"
LIBS_ROOT="${LIBS_ROOT:-${BRIDGEVLA_FINETUNE_ROOT}/bridgevla/libs}"
INSTALL="${INSTALL:-0}"
DOWNLOAD_COPPELIASIM="${DOWNLOAD_COPPELIASIM:-0}"
PIP_INSTALL_EDITABLE="${PIP_INSTALL_EDITABLE:-0}"
BUILD_PYREP="${BUILD_PYREP:-0}"

RL_BENCH_REPO="${RL_BENCH_REPO:-https://github.com/buttomnutstoast/RLBench.git}"
RL_BENCH_COMMIT="${RL_BENCH_COMMIT:-587a6a0e6dc8cd36612a208724eb275fe8cb4470}"
PYREP_REPO="${PYREP_REPO:-https://github.com/stepjam/PyRep.git}"
PYREP_COMMIT="${PYREP_COMMIT:-231a1ac6b0a179cff53c1d403d379260b9f05f2f}"
COPPELIASIM_URL="${COPPELIASIM_URL:-https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz}"
COPPELIASIM_ARCHIVE="${COPPELIASIM_ARCHIVE:-${BRIDGEVLA_FINETUNE_ROOT}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz}"
COPPELIASIM_DIR="${COPPELIASIM_DIR:-${BRIDGEVLA_FINETUNE_ROOT}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"

if [[ ! -d "${COPPELIASIM_DIR}" ]]; then
  for candidate in \
    "${BRIDGEVLA_FINETUNE_ROOT}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" \
    /root/autodl-tmp/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 \
    /root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04; do
    if [[ -d "${candidate}" ]]; then
      COPPELIASIM_DIR="${candidate}"
      break
    fi
  done
fi

cd /root/autodl-tmp/openvla-oft

echo "RLBench setup helper"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  BRIDGEVLA_FINETUNE_ROOT=${BRIDGEVLA_FINETUNE_ROOT}"
echo "  LIBS_ROOT=${LIBS_ROOT}"
echo "  INSTALL=${INSTALL}"
echo "  DOWNLOAD_COPPELIASIM=${DOWNLOAD_COPPELIASIM}"
echo "  PIP_INSTALL_EDITABLE=${PIP_INSTALL_EDITABLE}"
echo "  BUILD_PYREP=${BUILD_PYREP}"
echo "  COPPELIASIM_DIR=${COPPELIASIM_DIR}"

mkdir -p "${LIBS_ROOT}"

clone_if_missing() {
  local name="$1"
  local repo="$2"
  local commit="$3"
  local dst="${LIBS_ROOT}/${name}"
  if [[ -d "${dst}/.git" ]]; then
    echo "[ok] ${name} already cloned at ${dst}"
    return
  fi
  if [[ "${INSTALL}" != "1" ]]; then
    echo "[missing] ${name}: ${dst}"
    return
  fi
  git clone "${repo}" "${dst}"
  git -C "${dst}" checkout "${commit}"
}

clone_if_missing "RLBench" "${RL_BENCH_REPO}" "${RL_BENCH_COMMIT}"
clone_if_missing "PyRep" "${PYREP_REPO}" "${PYREP_COMMIT}"

if [[ "${DOWNLOAD_COPPELIASIM}" == "1" && ! -d "${COPPELIASIM_DIR}" ]]; then
  echo "[download] CoppeliaSim -> ${COPPELIASIM_ARCHIVE}"
  curl -L "${COPPELIASIM_URL}" -o "${COPPELIASIM_ARCHIVE}"
  mkdir -p "$(dirname "${COPPELIASIM_DIR}")"
  tar -xf "${COPPELIASIM_ARCHIVE}" -C "$(dirname "${COPPELIASIM_DIR}")"
elif [[ -d "${COPPELIASIM_DIR}" ]]; then
  echo "[ok] CoppeliaSim found at ${COPPELIASIM_DIR}"
else
  echo "[missing] CoppeliaSim: ${COPPELIASIM_DIR}"
fi

if [[ -d "${COPPELIASIM_DIR}" ]]; then
  export COPPELIASIM_ROOT="${COPPELIASIM_DIR}"
  export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-${COPPELIASIM_ROOT}}"
fi

if [[ "${INSTALL}" == "1" && "${PIP_INSTALL_EDITABLE}" == "1" ]]; then
  # Use --no-deps to avoid upgrading core packages such as numpy/torch in the
  # DepthVLA environment. The source_rlbench_env.sh PYTHONPATH route is usually
  # enough for conversion scripts, so editable pip installs are optional.
  "${PYTHON_BIN}" -m pip install --no-deps -e "${LIBS_ROOT}/YARR"
  "${PYTHON_BIN}" -m pip install --no-deps -e "${LIBS_ROOT}/peract_colab"
  if [[ -d "${LIBS_ROOT}/RLBench" ]]; then
    "${PYTHON_BIN}" -m pip install --no-deps -e "${LIBS_ROOT}/RLBench"
  fi
  if [[ -d "${LIBS_ROOT}/PyRep" ]]; then
    "${PYTHON_BIN}" -m pip install --no-deps -e "${LIBS_ROOT}/PyRep"
  fi
fi

if [[ "${BUILD_PYREP}" == "1" ]]; then
  if [[ ! -d "${COPPELIASIM_DIR}" ]]; then
    echo "[error] BUILD_PYREP=1 requires CoppeliaSim at ${COPPELIASIM_DIR}" >&2
    exit 2
  fi
  if [[ ! -d "${LIBS_ROOT}/PyRep" ]]; then
    echo "[error] BUILD_PYREP=1 requires PyRep source at ${LIBS_ROOT}/PyRep" >&2
    exit 2
  fi
  echo "[build] PyRep CFFI extension with COPPELIASIM_ROOT=${COPPELIASIM_ROOT}"
  (cd "${LIBS_ROOT}/PyRep" && "${PYTHON_BIN}" setup.py build_ext --inplace)
fi

echo
echo "To use the environment in the current shell:"
echo "  source experiments/robot/rlbench/source_rlbench_env.sh"
echo
echo "Current package check:"
source experiments/robot/rlbench/source_rlbench_env.sh
"${PYTHON_BIN}" experiments/robot/rlbench/check_rlbench_env.py
