#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter
PY=/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python
SRC="${ROOT}/src"
MJLAB_ROOT=/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab
OUT="${ROOT}/outputs/task044"
TASK="${TASK:-Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-Fast1p6}"
CKPT="${CKPT:-${OUT}/hidden_fault_train/logs_model5349_hidden_fault_env1024_iter25_seed4400301/model_24.pt}"
TAG="${TAG:-model5349_hidden_fault_env1024_iter25_model24_actionstats}"
SEED="${SEED:-4400801}"
NUM_ENVS="${NUM_ENVS:-1024}"
STEPS="${STEPS:-360}"
VX="${VX:-1.6}"
JOINT="${JOINT:-left_knee_joint}"
WINDOW_S="${WINDOW_S:-0.5}"
MEMORY_LATENT_SCALE="${MEMORY_LATENT_SCALE:-1.0}"
EVAL_DIR="${OUT}/hidden_fault_eval"
SUMMARY_DIR="${OUT}/action_influence_summary"
LOG_DIR="${OUT}/logs"

mkdir -p "${EVAL_DIR}" "${SUMMARY_DIR}" "${LOG_DIR}"

run_eval() {
  local mode="$1"
  local label="$2"
  local json="${EVAL_DIR}/${TAG}_${label}_${JOINT}_vx1p6_seed${SEED}.json"
  local log="${LOG_DIR}/${TAG}_${label}_${JOINT}_vx1p6_seed${SEED}.log"
  PYTHONPATH="${ROOT}:${SRC}:${MJLAB_ROOT}" "${PY}" -m h200_locomotion_lab.tools.task044_hidden_fault_eval \
    --task "${TASK}" \
    --checkpoint "${CKPT}" \
    --output-json "${json}" \
    --num-envs "${NUM_ENVS}" \
    --steps "${STEPS}" \
    --seed "${SEED}" \
    --device cuda:0 \
    --lin-vel-x "${VX}" \
    --lin-vel-y 0.0 \
    --ang-vel-z 0.0 \
    --dynamic-dead-joint "${JOINT}" \
    --dynamic-onset-s 0.0 \
    --dynamic-recovery-s 2.0 \
    --final-window-s "${WINDOW_S}" \
    --memory-latent-scale "${MEMORY_LATENT_SCALE}" \
    --memory-ablation-mode "${mode}" \
    >"${log}" 2>&1
  printf '%s\n' "${json}"
}

NORMAL_JSON="$(run_eval none none)"
ZERO_JSON="$(run_eval zero_txl_residual zero)"
STATELESS_JSON="$(run_eval stateless_txl_memory stateless)"
SUMMARY_JSON="${SUMMARY_DIR}/${TAG}_${JOINT}_vx1p6_seed${SEED}.json"
SUMMARY_LOG="${LOG_DIR}/${TAG}_action_summary_${JOINT}_vx1p6_seed${SEED}.log"
TRIPLET_DIR="${OUT}/triplet_summary"
TRIPLET_JSON="${TRIPLET_DIR}/${TAG}_triplet_${JOINT}_vx1p6_seed${SEED}.json"
TRIPLET_LOG="${LOG_DIR}/${TAG}_triplet_summary_${JOINT}_vx1p6_seed${SEED}.log"

PYTHONPATH="${ROOT}:${SRC}:${MJLAB_ROOT}" "${PY}" -m h200_locomotion_lab.tools.task044_action_influence_summary \
  --normal-json "${NORMAL_JSON}" \
  --zero-residual-json "${ZERO_JSON}" \
  --stateless-json "${STATELESS_JSON}" \
  --metric-scope final_trial_window \
  --output-json "${SUMMARY_JSON}" \
  >"${SUMMARY_LOG}" 2>&1

mkdir -p "${TRIPLET_DIR}"
PYTHONPATH="${ROOT}:${SRC}:${MJLAB_ROOT}" "${PY}" -m h200_locomotion_lab.tools.task044_memory_required_triplet_summary \
  --normal-json "${NORMAL_JSON}" \
  --zero-residual-json "${ZERO_JSON}" \
  --stateless-json "${STATELESS_JSON}" \
  --confirm-hidden-fault-labels \
  --metric-scope final_trial_window \
  --output-json "${TRIPLET_JSON}" \
  >"${TRIPLET_LOG}" 2>&1

"${PY}" - "${SUMMARY_JSON}" "${TRIPLET_JSON}" <<'PY'
import json
import sys

path = sys.argv[1]
triplet_path = sys.argv[2]
data = json.load(open(path, encoding="utf-8"))
triplet = json.load(open(triplet_path, encoding="utf-8"))
contract = data["task044_action_influence_contract"]
print(json.dumps({
    "summary_json": path,
    "triplet_json": triplet_path,
    "action_influence_detected": data["action_influence_detected"],
    "task044_memory_required_pass": triplet.get("task044_memory_required_pass"),
    "triplet_failure_reasons": triplet.get("failure_reasons"),
    "failure_reasons": data["failure_reasons"],
    "zero_mean_abs_l1_delta": contract["zero_residual_delta"].get("mean_abs_l1_delta"),
    "zero_mean_l2_delta": contract["zero_residual_delta"].get("mean_l2_delta"),
    "stateless_mean_abs_l1_delta": contract["stateless_memory_delta"].get("mean_abs_l1_delta"),
    "stateless_mean_l2_delta": contract["stateless_memory_delta"].get("mean_l2_delta"),
}, indent=2, sort_keys=True))
PY
