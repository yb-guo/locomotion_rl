#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat >&2 <<'USAGE'
usage: bash task036_launch_policy_train.sh

Environment overrides:
  CONSUMER, TASK, GPU_ID, NUM_ENVS, MAX_ITER, SAVE_INTERVAL, SEED, RUN_NAME,
  EXPERIMENT, LOAD_RUN, LOAD_CHECKPOINT, WARMSTART_CHECKPOINT, LEARNING_RATE,
  ENTROPY_COEF, RESUME, DRY_RUN
USAGE
  exit 0
fi

ROOT="${ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter}"
OUT="${OUT:-${ADAPTER_ROOT}/outputs/task036/policy_train}"
PY="${PY:-/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python}"
CONSUMER="${CONSUMER:-adapt}"
GPU_ID="${GPU_ID:-0}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITER="${MAX_ITER:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
SEED="${SEED:-3603601}"
EXPERIMENT="${EXPERIMENT:-g1_gripper_velocity_task036_policy_quality_train}"
RUN_NAME="${RUN_NAME:-036_${CONSUMER}_env${NUM_ENVS}_iter${MAX_ITER}_gpu${GPU_ID}_seed${SEED}}"
SESSION="${SESSION:-task036_${CONSUMER}_gpu${GPU_ID}_seed${SEED}}"
LEARNING_RATE="${LEARNING_RATE:-0.000003}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0003}"
RESUME="${RESUME:-}"

case "${CONSUMER}" in
  adapt)
    TASK="${TASK:-Unitree-G1-Gripper-Flat-Task036-AdaptK4-Fast2p0}"
    RESUME="${RESUME:-1}"
    LOAD_RUN="${LOAD_RUN:-task036_model5349_adapt_warmstart}"
    LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-model_5349.pt}"
    WARMSTART_CHECKPOINT="${WARMSTART_CHECKPOINT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task030_dynamic_004_train/2026-05-21_17-35-22_005_kneehiproll_vx2p0_from5320_env8192_iter30_gpu1_seed30750/model_5349.pt}"
    ;;
  gru)
    TASK="${TASK:-Unitree-G1-Gripper-Flat-Task036-GruK4-Fast2p0}"
    ;;
  token)
    TASK="${TASK:-Unitree-G1-Gripper-Flat-Task036-TokenK4-Fast2p0}"
    ;;
  *)
    echo "unknown CONSUMER=${CONSUMER}" >&2
    exit 2
    ;;
esac
RESUME="${RESUME:-0}"

mkdir -p "${OUT}"

resume_args=()
if [[ "${RESUME}" == "1" ]]; then
  if [[ -z "${LOAD_RUN:-}" || -z "${LOAD_CHECKPOINT:-}" || -z "${WARMSTART_CHECKPOINT:-}" ]]; then
    echo "resume requires LOAD_RUN, LOAD_CHECKPOINT, and WARMSTART_CHECKPOINT" >&2
    exit 2
  fi
  if [[ ! -f "${WARMSTART_CHECKPOINT}" ]]; then
    echo "missing WARMSTART_CHECKPOINT=${WARMSTART_CHECKPOINT}" >&2
    exit 3
  fi
  mkdir -p "${ROOT}/logs/rsl_rl/${EXPERIMENT}/${LOAD_RUN}"
  ln -sfn "${WARMSTART_CHECKPOINT}" "${ROOT}/logs/rsl_rl/${EXPERIMENT}/${LOAD_RUN}/${LOAD_CHECKPOINT}"
  resume_args=(
    "--agent.resume=True"
    "--agent.load-run=${LOAD_RUN}"
    "--agent.load-checkpoint=${LOAD_CHECKPOINT}"
  )
fi

CMD="cd '${ROOT}' && PYTHONPATH='${ADAPTER_ROOT}/src:/tmp/task029_ipython_stub:/tmp/task029_pydeps:.' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true '${PY}' scripts/train.py '${TASK}' --gpu-ids=[${GPU_ID}] --env.scene.num-envs=${NUM_ENVS} --agent.max-iterations=${MAX_ITER} --agent.save-interval=${SAVE_INTERVAL} --agent.experiment-name='${EXPERIMENT}' --agent.run-name='${RUN_NAME}' --agent.logger=tensorboard --agent.upload-model=False --agent.seed=${SEED} ${resume_args[*]} --agent.algorithm.learning-rate=${LEARNING_RATE} --agent.algorithm.entropy-coef=${ENTROPY_COEF} 2>&1 | tee '${OUT}/${RUN_NAME}.stdout.log'"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "tmux new-session -d -s '${SESSION}' \"${CMD}\""
  exit 0
fi

tmux new-session -d -s "${SESSION}" "${CMD}"
tmux list-sessions | grep "${SESSION}"
