#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat >&2 <<'USAGE'
usage: bash task034_launch_right_knee_curriculum.sh

Environment overrides:
  TASK, STAGE, GPU_ID, NUM_ENVS, MAX_ITER, SAVE_INTERVAL, SEED, LOAD_RUN,
  LOAD_CHECKPOINT, WARMSTART_CHECKPOINT, LEARNING_RATE, ENTROPY_COEF, DRY_RUN
USAGE
  exit 0
fi

ROOT="${ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab}"
OUT="${OUT:-/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task034/right_knee_curriculum_train}"
PY="${PY:-/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python}"
STAGE="${STAGE:-mixed}"
TASK="${TASK:-Unitree-G1-Gripper-Flat-Task034-RightKneeMixed-FrozenBase-Fast2p0}"
EXPERIMENT="${EXPERIMENT:-g1_gripper_velocity_task034_rightknee_frozenbase_train}"
GPU_ID="${GPU_ID:-1}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITER="${MAX_ITER:-30}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
SEED="${SEED:-3403401}"
RUN_NAME="${RUN_NAME:-034_${STAGE}_rightknee_from_task033_model5378_env${NUM_ENVS}_iter${MAX_ITER}_gpu${GPU_ID}_seed${SEED}}"
SESSION="${SESSION:-task034_${STAGE}_rightknee_gpu${GPU_ID}_seed${SEED}}"
LOAD_RUN="${LOAD_RUN:-task034_task033_model5378_warmstart}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-model_5378.pt}"
WARMSTART_CHECKPOINT="${WARMSTART_CHECKPOINT:-${ROOT}/logs/rsl_rl/g1_gripper_velocity_task033_stackmlp_k4_frozenbase_focused/2026-05-28_12-40-56_033_frozenbase_focused_from5349_env8192_iter30_gpu1_seed3303362_lr1e5/model_5378.pt}"
LEARNING_RATE="${LEARNING_RATE:-0.000005}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0005}"

mkdir -p "${OUT}"
mkdir -p "${ROOT}/logs/rsl_rl/${EXPERIMENT}/${LOAD_RUN}"
if [[ ! -f "${WARMSTART_CHECKPOINT}" ]]; then
  echo "missing WARMSTART_CHECKPOINT=${WARMSTART_CHECKPOINT}" >&2
  exit 3
fi
ln -sfn "${WARMSTART_CHECKPOINT}" "${ROOT}/logs/rsl_rl/${EXPERIMENT}/${LOAD_RUN}/${LOAD_CHECKPOINT}"

CMD="cd '${ROOT}' && PYTHONPATH='/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/src:/tmp/task029_ipython_stub:/tmp/task029_pydeps:.' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true '${PY}' scripts/train.py '${TASK}' --gpu-ids=[${GPU_ID}] --env.scene.num-envs=${NUM_ENVS} --agent.max-iterations=${MAX_ITER} --agent.save-interval=${SAVE_INTERVAL} --agent.experiment-name='${EXPERIMENT}' --agent.run-name='${RUN_NAME}' --agent.logger=tensorboard --agent.upload-model=False --agent.seed=${SEED} --agent.resume=True --agent.load-run='${LOAD_RUN}' --agent.load-checkpoint='${LOAD_CHECKPOINT}' --agent.algorithm.learning-rate=${LEARNING_RATE} --agent.algorithm.entropy-coef=${ENTROPY_COEF} 2>&1 | tee '${OUT}/${RUN_NAME}.stdout.log'"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "tmux new-session -d -s '${SESSION}' \"${CMD}\""
  exit 0
fi

tmux new-session -d -s "${SESSION}" "${CMD}"
tmux ls | grep "${SESSION}"
