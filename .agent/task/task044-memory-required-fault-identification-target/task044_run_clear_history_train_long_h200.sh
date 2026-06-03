#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter
MJLAB_ROOT=/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab
PY=/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python
SRC="${ROOT}/src"
OUT="${ROOT}/outputs/task044/hidden_fault_train"
TASK="${TASK:-Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-Fast1p6}"
CKPT="${CKPT:-${OUT}/logs_model5349_hidden_fault_env1024_iter25_seed4400301/model_24.pt}"
SEED="${SEED:-4401101}"
ITERS="${ITERS:-50}"
NUM_ENVS="${NUM_ENVS:-1024}"
DEVICE="${DEVICE:-cuda:0}"
MEMORY_LATENT_SCALE="${MEMORY_LATENT_SCALE:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-}"
DESIRED_KL="${DESIRED_KL:-}"
FAULT_AUX_LOSS_WEIGHT="${FAULT_AUX_LOSS_WEIGHT:-0.0}"
FAULT_AUX_NUM_CLASSES="${FAULT_AUX_NUM_CLASSES:-0}"
FAULT_AUX_MAX_TRIAL_STEP="${FAULT_AUX_MAX_TRIAL_STEP:--1}"
FAULT_AUX_MIN_TRIAL_INDEX="${FAULT_AUX_MIN_TRIAL_INDEX:-0}"
ACTOR_TRAINABLE_SCOPE="${ACTOR_TRAINABLE_SCOPE:-txl_residual_and_mlp_memory_input}"
SCALE_TAG="${MEMORY_LATENT_SCALE/./p}"
RUN_TAG="${RUN_TAG:-clear_history_env${NUM_ENVS}_iter${ITERS}_scale${SCALE_TAG}_seed${SEED}}"
JSON="${OUT}/train_${RUN_TAG}.json"
LOG_DIR="${OUT}/logs_${RUN_TAG}"
STDOUT_LOG="${OUT}/${RUN_TAG}.stdout.log"

mkdir -p "${OUT}"

EXTRA_ARGS=()
if [[ -n "${LEARNING_RATE}" ]]; then
  EXTRA_ARGS+=(--learning-rate "${LEARNING_RATE}")
fi
if [[ -n "${DESIRED_KL}" ]]; then
  EXTRA_ARGS+=(--desired-kl "${DESIRED_KL}")
fi

PYTHONPATH="${ROOT}:${SRC}:${MJLAB_ROOT}" "${PY}" -m h200_locomotion_lab.tools.task044_hidden_fault_train \
  --task "${TASK}" \
  --output-json "${JSON}" \
  --log-dir "${LOG_DIR}" \
  --num-envs "${NUM_ENVS}" \
  --rollout-steps 24 \
  --iterations "${ITERS}" \
  --save-interval 10 \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --num-mini-batches 4 \
  --resume-checkpoint "${CKPT}" \
  --memory-latent-scale "${MEMORY_LATENT_SCALE}" \
  "${EXTRA_ARGS[@]}" \
  --task044-fault-aux-loss-weight "${FAULT_AUX_LOSS_WEIGHT}" \
  --task044-fault-aux-num-classes "${FAULT_AUX_NUM_CLASSES}" \
  --task044-fault-aux-max-trial-step "${FAULT_AUX_MAX_TRIAL_STEP}" \
  --task044-fault-aux-min-trial-index "${FAULT_AUX_MIN_TRIAL_INDEX}" \
  --actor-trainable-scope "${ACTOR_TRAINABLE_SCOPE}" \
  --run-name "${RUN_TAG}" \
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
    "wall_time_s": data.get("wall_time_s"),
}, indent=2, sort_keys=True))
PY
