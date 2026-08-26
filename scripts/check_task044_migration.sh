#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITREE_MJLAB_ROOT="${UNITREE_MJLAB_ROOT:-${REPO_ROOT}/.external/unitree_rl_mjlab}"
TASK_CFG_ROOT="${UNITREE_MJLAB_ROOT}/src/tasks/velocity/config/g1_gripper"
ASSET_ROOT="${UNITREE_MJLAB_ROOT}/src/assets/robots/unitree_g1_gripper"
CHECKPOINT="${TASK044_CHECKPOINT:-}"
REQUIRE_CHECKPOINT="${REQUIRE_TASK044_CHECKPOINT:-0}"
MISSING=()

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    MISSING+=("${label}: ${path}")
  fi
}

require_pattern() {
  local path="$1"
  local pattern="$2"
  local label="$3"
  if [[ ! -f "${path}" ]] || ! rg --fixed-strings --quiet "${pattern}" "${path}"; then
    MISSING+=("${label}: ${pattern} in ${path}")
  fi
}

require_file "${ASSET_ROOT}/g1_gripper_constants.py" "Task028 gripper constants"
require_file "${ASSET_ROOT}/xmls/g1_gripper.xml" "Task028 gripper MJCF"
require_file "${TASK_CFG_ROOT}/rl_cfg.py" "Task028 gripper RL config"
require_pattern "${TASK_CFG_ROOT}/env_cfgs.py" \
  "def _add_motor_failure_stage" \
  "Task029 motor-failure environment base"
require_pattern "${TASK_CFG_ROOT}/env_cfgs.py" \
  "unitree_g1_gripper_flat_dynamic_failure_fast1p6_env_cfg" \
  "Task030 dynamic-failure environment"
require_pattern "${TASK_CFG_ROOT}/env_cfgs.py" \
  "unitree_g1_gripper_flat_task031_focused_deadgrid_env_cfg" \
  "Task031 unified-speed/dead-grid environment"
require_pattern "${TASK_CFG_ROOT}/__init__.py" \
  "Task044TrueTxlMemoryK160ClearHistoryRunner" \
  "Task044 runner registration"
require_pattern "${TASK_CFG_ROOT}/__init__.py" \
  "Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-Fast1p6" \
  "Task044 task registration"
require_pattern "${REPO_ROOT}/src/h200_locomotion_lab/training/rsl_history_wrapper.py" \
  "class Task044TrueTxlMemoryK160ClearHistoryRunner" \
  "local true-TXL runner"

if [[ -n "${CHECKPOINT}" ]]; then
  require_file "${CHECKPOINT}" "Task044 checkpoint"
elif [[ "${REQUIRE_CHECKPOINT}" == "1" ]]; then
  MISSING+=("Task044 checkpoint: set TASK044_CHECKPOINT")
fi

if (( ${#MISSING[@]} > 0 )); then
  echo "Task044 custom MJLab migration is incomplete:" >&2
  for item in "${MISSING[@]}"; do
    echo "  - ${item}" >&2
  done
  echo >&2
  echo "The official Unitree-G1-Flat smoke may still run; do not treat it as the 31-action custom algorithm." >&2
  exit 2
fi

echo "Task044 custom source registration is present."
if [[ -n "${CHECKPOINT}" ]]; then
  echo "Checkpoint present: ${CHECKPOINT}"
else
  echo "No checkpoint requested; this verifies source registration only."
fi
