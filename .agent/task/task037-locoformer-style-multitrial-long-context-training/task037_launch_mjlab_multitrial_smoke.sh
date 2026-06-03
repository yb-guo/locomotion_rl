#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat >&2 <<'USAGE'
usage: bash task037_launch_mjlab_multitrial_smoke.sh

Environment overrides:
  TASK, GPU_ID, NUM_ENVS, MAX_ITER, SAVE_INTERVAL, SEED, RUN_NAME, EXPERIMENT,
  ROOT, ADAPTER_ROOT, OUT, PY, DRY_RUN
USAGE
  exit 0
fi

ROOT="${ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter}"
OUT="${OUT:-${ADAPTER_ROOT}/outputs/task037/mjlab_multitrial_smoke}"
PY="${PY:-/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python}"
TASK="${TASK:-Unitree-G1-Gripper-Flat-Task037-BufferOnlyK4-AutoReset-Fast2p0}"
GPU_ID="${GPU_ID:-0}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITER="${MAX_ITER:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
SEED="${SEED:-3700202}"
EXPERIMENT="${EXPERIMENT:-g1_gripper_velocity_task037_multitrial_smoke}"
RUN_NAME="${RUN_NAME:-037_bufferonly_multitrial_env${NUM_ENVS}_iter${MAX_ITER}_gpu${GPU_ID}_seed${SEED}}"
SESSION="${SESSION:-task037_multitrial_env${NUM_ENVS}_gpu${GPU_ID}_seed${SEED}}"

mkdir -p "${OUT}"

CMD="cd '${ROOT}' && PYTHONPATH='${ADAPTER_ROOT}/src:/tmp/task029_ipython_stub:/tmp/task029_pydeps:.' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true '${PY}' scripts/train.py '${TASK}' --gpu-ids=[${GPU_ID}] --env.scene.num-envs=${NUM_ENVS} --agent.max-iterations=${MAX_ITER} --agent.save-interval=${SAVE_INTERVAL} --agent.experiment-name='${EXPERIMENT}' --agent.run-name='${RUN_NAME}' --agent.logger=tensorboard --agent.upload-model=False --agent.seed=${SEED} 2>&1 | tee '${OUT}/${RUN_NAME}.stdout.log'"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "tmux new-session -d -s '${SESSION}' \"${CMD}\""
  exit 0
fi

tmux new-session -d -s "${SESSION}" "${CMD}"
tmux list-sessions | grep "${SESSION}"
