#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter
MJLAB_ROOT=/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab
PY=/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python
SRC="${ROOT}/src"
OUT="${ROOT}/outputs/task044/hidden_fault_train"
TASK="${TASK:-Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-Fast1p6}"
CKPT="${OUT}/logs_model5349_hidden_fault_env1024_iter25_seed4400301/model_24.pt"
SEED="${SEED:-4401001}"
FAULT_AUX_LOSS_WEIGHT="${FAULT_AUX_LOSS_WEIGHT:-0.0}"
FAULT_AUX_NUM_CLASSES="${FAULT_AUX_NUM_CLASSES:-0}"
FAULT_AUX_MAX_TRIAL_STEP="${FAULT_AUX_MAX_TRIAL_STEP:--1}"
FAULT_AUX_MIN_TRIAL_INDEX="${FAULT_AUX_MIN_TRIAL_INDEX:-0}"
JSON="${OUT}/train_clear_history_smoke_env64_iter1_seed${SEED}.json"
LOG_DIR="${OUT}/logs_clear_history_smoke_env64_iter1_seed${SEED}"
STDOUT_LOG="${OUT}/clear_history_smoke_env64_iter1_seed${SEED}.stdout.log"

mkdir -p "${OUT}"

PYTHONPATH="${ROOT}:${SRC}:${MJLAB_ROOT}" "${PY}" -m h200_locomotion_lab.tools.task044_hidden_fault_train \
  --task "${TASK}" \
  --output-json "${JSON}" \
  --log-dir "${LOG_DIR}" \
  --num-envs 64 \
  --rollout-steps 24 \
  --iterations 1 \
  --save-interval 1 \
  --seed "${SEED}" \
  --device cuda:0 \
  --num-mini-batches 4 \
  --resume-checkpoint "${CKPT}" \
  --task044-fault-aux-loss-weight "${FAULT_AUX_LOSS_WEIGHT}" \
  --task044-fault-aux-num-classes "${FAULT_AUX_NUM_CLASSES}" \
  --task044-fault-aux-max-trial-step "${FAULT_AUX_MAX_TRIAL_STEP}" \
  --task044-fault-aux-min-trial-index "${FAULT_AUX_MIN_TRIAL_INDEX}" \
  --run-name "clear_history_smoke_env64_iter1_seed${SEED}" \
  >"${STDOUT_LOG}" 2>&1

"${PY}" - "${JSON}" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
print(json.dumps({
    "json": path,
    "task": data.get("task"),
    "train_pipeline_pass": data.get("train_pipeline_pass"),
    "task044_train_pipeline_pass": data.get("task044_train_pipeline_pass"),
    "runner_cls": data.get("runner_cls"),
    "algorithm_class": data.get("algorithm_class"),
    "checkpoint": data.get("checkpoint"),
    "task044_fault_aux_loss_weight": data.get("task044_fault_aux_loss_weight"),
    "task044_fault_aux_num_classes": data.get("task044_fault_aux_num_classes"),
    "task044_fault_aux_max_trial_step": data.get("task044_fault_aux_max_trial_step"),
    "task044_fault_aux_min_trial_index": data.get("task044_fault_aux_min_trial_index"),
    "algorithm_debug": data.get("algorithm_debug"),
    "failure_reasons": data.get("failure_reasons"),
}, indent=2, sort_keys=True))
PY
