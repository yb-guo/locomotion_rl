#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat >&2 <<'USAGE'
usage: bash task037_launch_clean_gait_prior.sh

Launch a Task037 clean-only K160 adaptation warmstart run on H200.

Environment overrides:
  ROOT, ADAPTER_ROOT, OUT, PY, TASK, GPU_ID, NUM_ENVS, MAX_ITER, SAVE_INTERVAL,
  SEED, RUN_NAME, SESSION, EXPERIMENT, LOAD_RUN, LOAD_CHECKPOINT,
  WARMSTART_CHECKPOINT, LEARNING_RATE, ENTROPY_COEF, DRY_RUN
USAGE
  exit 0
fi

ROOT="${ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter}"
OUT="${OUT:-${ADAPTER_ROOT}/outputs/task037/clean_gait_prior}"
PY="${PY:-/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python}"
TASK="${TASK:-Unitree-G1-Gripper-Flat-Task037-AdaptK160-CleanUnified-Fast2p0}"
GPU_ID="${GPU_ID:-0}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITER="${MAX_ITER:-60}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
SEED="${SEED:-3700702}"
EXPERIMENT="${EXPERIMENT:-g1_gripper_velocity_task037_clean_gait_prior_train}"
RUN_NAME="${RUN_NAME:-037_adapt_k160_clean_from_adaptk4_env${NUM_ENVS}_iter${MAX_ITER}_gpu${GPU_ID}_seed${SEED}}"
SESSION="${SESSION:-task037_clean_gait_prior_gpu${GPU_ID}_seed${SEED}}"
LOAD_RUN="${LOAD_RUN:-task037_adaptk4_model5408_k160_warmstart}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-model_5408.pt}"
WARMSTART_CHECKPOINT="${WARMSTART_CHECKPOINT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task036_policy_quality_train/2026-05-28_23-10-53_036_adapt_k4_warmstart_env8192_iter60_gpu0_seed3603630/model_5408.pt}"
LEARNING_RATE="${LEARNING_RATE:-0.000003}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0003}"

if [[ ! -f "${WARMSTART_CHECKPOINT}" ]]; then
  echo "missing WARMSTART_CHECKPOINT=${WARMSTART_CHECKPOINT}" >&2
  exit 3
fi

mkdir -p "${OUT}"
mkdir -p "${ROOT}/logs/rsl_rl/${EXPERIMENT}/${LOAD_RUN}"
ln -sfn "${WARMSTART_CHECKPOINT}" "${ROOT}/logs/rsl_rl/${EXPERIMENT}/${LOAD_RUN}/${LOAD_CHECKPOINT}"

CMD="cd '${ROOT}' && PYTHONPATH='${ADAPTER_ROOT}/src:/tmp/task029_ipython_stub:/tmp/task029_pydeps:.' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true '${PY}' scripts/train.py '${TASK}' --gpu-ids=[${GPU_ID}] --env.scene.num-envs=${NUM_ENVS} --agent.max-iterations=${MAX_ITER} --agent.save-interval=${SAVE_INTERVAL} --agent.experiment-name='${EXPERIMENT}' --agent.run-name='${RUN_NAME}' --agent.logger=tensorboard --agent.upload-model=False --agent.seed=${SEED} --agent.resume=True --agent.load-run='${LOAD_RUN}' --agent.load-checkpoint='${LOAD_CHECKPOINT}' --agent.algorithm.learning-rate=${LEARNING_RATE} --agent.algorithm.entropy-coef=${ENTROPY_COEF} 2>&1 | tee '${OUT}/${RUN_NAME}.stdout.log'"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "tmux new-session -d -s '${SESSION}' \"${CMD}\""
  exit 0
fi

tmux new-session -d -s "${SESSION}" "${CMD}"
tmux list-sessions | grep "${SESSION}"
