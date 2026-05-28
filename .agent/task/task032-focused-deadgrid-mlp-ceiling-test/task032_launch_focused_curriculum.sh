#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat >&2 <<'USAGE'
usage: bash task032_launch_focused_curriculum.sh

Environment overrides:
  TASK, STAGE, GPU_ID, NUM_ENVS, MAX_ITER, SAVE_INTERVAL, SEED, LOAD_RUN,
  LOAD_CHECKPOINT, WARMSTART_CHECKPOINT, LEARNING_RATE, ENTROPY_COEF, DRY_RUN
USAGE
  exit 0
fi

ROOT="${ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab}"
OUT="${OUT:-/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task032/focused_curriculum_train}"
PY="${PY:-/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python}"
STAGE="${STAGE:-weak}"
TASK="${TASK:-Unitree-G1-Gripper-Flat-Task032-WeakFocused-Fast2p0}"
EXPERIMENT="${EXPERIMENT:-g1_gripper_velocity_task032_mlp_ceiling_train}"
GPU_ID="${GPU_ID:-1}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITER="${MAX_ITER:-40}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
SEED="${SEED:-3203203}"
RUN_NAME="${RUN_NAME:-032_${STAGE}_focused_from5349_env${NUM_ENVS}_iter${MAX_ITER}_gpu${GPU_ID}_seed${SEED}}"
SESSION="${SESSION:-task032_${STAGE}_focused_gpu${GPU_ID}_seed${SEED}}"
LOAD_RUN="${LOAD_RUN:-task032_model5349_warmstart}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-model_5349.pt}"
WARMSTART_CHECKPOINT="${WARMSTART_CHECKPOINT:-${ROOT}/logs/rsl_rl/g1_gripper_velocity_task030_dynamic_004_train/2026-05-21_17-35-22_005_kneehiproll_vx2p0_from5320_env8192_iter30_gpu1_seed30750/model_5349.pt}"
LEARNING_RATE="${LEARNING_RATE:-0.00008}"
ENTROPY_COEF="${ENTROPY_COEF:-0.001}"

mkdir -p "${OUT}"
mkdir -p "${ROOT}/logs/rsl_rl/${EXPERIMENT}/${LOAD_RUN}"
if [[ ! -f "${WARMSTART_CHECKPOINT}" ]]; then
  echo "missing WARMSTART_CHECKPOINT=${WARMSTART_CHECKPOINT}" >&2
  exit 3
fi
ln -sfn "${WARMSTART_CHECKPOINT}" "${ROOT}/logs/rsl_rl/${EXPERIMENT}/${LOAD_RUN}/${LOAD_CHECKPOINT}"

CMD="cd '${ROOT}' && PYTHONPATH=/tmp/task029_ipython_stub:/tmp/task029_pydeps:. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true '${PY}' scripts/train.py '${TASK}' --gpu-ids=[${GPU_ID}] --env.scene.num-envs=${NUM_ENVS} --agent.max-iterations=${MAX_ITER} --agent.save-interval=${SAVE_INTERVAL} --agent.experiment-name='${EXPERIMENT}' --agent.run-name='${RUN_NAME}' --agent.logger=tensorboard --agent.upload-model=False --agent.seed=${SEED} --agent.resume=True --agent.load-run='${LOAD_RUN}' --agent.load-checkpoint='${LOAD_CHECKPOINT}' --agent.algorithm.learning-rate=${LEARNING_RATE} --agent.algorithm.entropy-coef=${ENTROPY_COEF} 2>&1 | tee '${OUT}/${RUN_NAME}.stdout.log'"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "tmux new-session -d -s '${SESSION}' \"${CMD}\""
  exit 0
fi

tmux new-session -d -s "${SESSION}" "${CMD}"
tmux ls | grep "${SESSION}"
