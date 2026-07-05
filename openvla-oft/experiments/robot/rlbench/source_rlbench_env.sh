#!/usr/bin/env bash
# Source this file before running RLBench conversion/evaluation commands.
#
#   source experiments/robot/rlbench/source_rlbench_env.sh
#
# It is intentionally side-effect-light: it only exports paths and does not
# install packages or start a display server.

export BRIDGEVLA_FINETUNE_ROOT="${BRIDGEVLA_FINETUNE_ROOT:-/root/autodl-tmp/BridgeVLA/finetune}"
export BRIDGEVLA_LIBS_ROOT="${BRIDGEVLA_LIBS_ROOT:-${BRIDGEVLA_FINETUNE_ROOT}/bridgevla/libs}"

prepend_pythonpath() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    case ":${PYTHONPATH:-}:" in
      *":${path}:"*) ;;
      *) export PYTHONPATH="${path}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
  fi
}

prepend_ld_library_path() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":${path}:"*) ;;
      *) export LD_LIBRARY_PATH="${path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
  fi
}

prepend_pythonpath "${BRIDGEVLA_FINETUNE_ROOT}"
prepend_pythonpath "${BRIDGEVLA_LIBS_ROOT}/RLBench"
prepend_pythonpath "${BRIDGEVLA_LIBS_ROOT}/PyRep"
prepend_pythonpath "${BRIDGEVLA_LIBS_ROOT}/YARR"
prepend_pythonpath "${BRIDGEVLA_LIBS_ROOT}/peract_colab"
prepend_pythonpath "${BRIDGEVLA_LIBS_ROOT}/peract"
prepend_pythonpath "${BRIDGEVLA_LIBS_ROOT}/point-renderer"

if [[ -z "${COPPELIASIM_ROOT:-}" ]]; then
  for candidate in \
    "${BRIDGEVLA_FINETUNE_ROOT}"/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 \
    /root/autodl-tmp/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 \
    /root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04; do
    if [[ -d "${candidate}" ]]; then
      export COPPELIASIM_ROOT="${candidate}"
      break
    fi
  done
fi

if [[ -n "${COPPELIASIM_ROOT:-}" ]]; then
  prepend_ld_library_path "${COPPELIASIM_ROOT}"
  # Force CoppeliaSim's Qt plugins. Some Python wheels such as opencv-python
  # set this to their own plugin directory, which can make CoppeliaSim fail to
  # locate the xcb platform plugin at launch time.
  export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
  export QT_PLUGIN_PATH="${COPPELIASIM_ROOT}"
fi

export DISPLAY="${DISPLAY:-:1.0}"
