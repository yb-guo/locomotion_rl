#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: bash task048_launch_reproduction_stage.sh STAGE

Stages:
  mlp-prior       Train a fresh MLP with the official command curriculum.
  mlp-speed-bins  Continue an MLP prior on clean 0.4/1.2/2.0 m/s bins.
  adaptk4    Warmstart AdaptK4 from an MLP checkpoint.
  adaptk160  Warmstart AdaptK160 from an AdaptK4 checkpoint.
  bridge     Convert an AdaptK160 checkpoint to the Task041 True-TXL shape.

Environment overrides:
  PROFILE=4090|historical, SOURCE_CHECKPOINT, ROOT, PROJECT_ROOT, PY, GPU_ID,
  NUM_ENVS, MAX_ITER, SAVE_INTERVAL, SEED, LEARNING_RATE, ENTROPY_COEF,
  EXPERIMENT, RUN_NAME, OUT, TARGET_CHECKPOINT, OUTPUT_JSON, DRY_RUN.

The historical profile is exact for the two migration-stage budgets. A fresh
MLP prior is a practical reconstruction because model_5349 came from a longer
manually gated curriculum.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || -z "${1:-}" ]]; then
  usage
  [[ -n "${1:-}" ]] && exit 0
  exit 2
fi

STAGE="$1"
PROFILE="${PROFILE:-4090}"
ROOT="${ROOT:-/home/xyzl/yubo/locomotion_rl/external/unitree_rl_mjlab}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/xyzl/yubo/locomotion_rl}"
PY="${PY:-${PROJECT_ROOT}/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
OUT="${OUT:-${PROJECT_ROOT}/outputs/task048}"

if [[ "${PROFILE}" != "4090" && "${PROFILE}" != "historical" ]]; then
  echo "PROFILE must be 4090 or historical, got ${PROFILE}" >&2
  exit 2
fi
if [[ ! -x "${PY}" ]]; then
  echo "missing Python executable: ${PY}" >&2
  exit 3
fi
if [[ ! -d "${ROOT}" ]]; then
  echo "missing Unitree MJLab root: ${ROOT}" >&2
  exit 3
fi

case "${STAGE}" in
  mlp-prior)
    TASK="Unitree-G1-Gripper-Flat-Task048-Mlp-OfficialCurriculum-Train"
    DEFAULT_NUM_ENVS=4096
    DEFAULT_MAX_ITER=1000
    DEFAULT_SAVE_INTERVAL=100
    DEFAULT_SEED=4800101
    DEFAULT_LEARNING_RATE=0.001
    DEFAULT_ENTROPY_COEF=0.01
    ;;
  mlp-speed-bins)
    TASK="Unitree-G1-Gripper-Flat-Task048-Mlp-CleanBins-Train"
    if [[ "${PROFILE}" == "historical" ]]; then
      DEFAULT_NUM_ENVS=8192
      DEFAULT_MAX_ITER=200
    else
      DEFAULT_NUM_ENVS=4096
      DEFAULT_MAX_ITER=400
    fi
    DEFAULT_SAVE_INTERVAL=50
    DEFAULT_SEED=4800102
    DEFAULT_LEARNING_RATE=0.00003
    DEFAULT_ENTROPY_COEF=0.001
    ;;
  adaptk4)
    TASK="Unitree-G1-Gripper-Flat-Task048-AdaptK4-CleanBins-Train"
    if [[ "${PROFILE}" == "historical" ]]; then
      DEFAULT_NUM_ENVS=8192
      DEFAULT_MAX_ITER=60
    else
      DEFAULT_NUM_ENVS=4096
      DEFAULT_MAX_ITER=120
    fi
    DEFAULT_SAVE_INTERVAL=10
    DEFAULT_SEED=3603630
    DEFAULT_LEARNING_RATE=0.000003
    DEFAULT_ENTROPY_COEF=0.0003
    ;;
  adaptk160)
    TASK="Unitree-G1-Gripper-Flat-Task048-AdaptK160-CleanBins"
    if [[ "${PROFILE}" == "historical" ]]; then
      DEFAULT_NUM_ENVS=8192
      DEFAULT_MAX_ITER=60
    else
      DEFAULT_NUM_ENVS=2048
      DEFAULT_MAX_ITER=240
    fi
    DEFAULT_SAVE_INTERVAL=10
    DEFAULT_SEED=3700705
    DEFAULT_LEARNING_RATE=0.000003
    DEFAULT_ENTROPY_COEF=0.0003
    ;;
  bridge)
    if [[ -z "${SOURCE_CHECKPOINT:-}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
      echo "bridge requires an existing SOURCE_CHECKPOINT" >&2
      exit 3
    fi
    TARGET_CHECKPOINT="${TARGET_CHECKPOINT:-${OUT}/bridge/model_task048_true_txl_bridge.pt}"
    OUTPUT_JSON="${OUTPUT_JSON:-${OUT}/bridge/model_task048_true_txl_bridge.json}"
    cmd=(
      "${PY}" -m h200_locomotion_lab.tools.task041_adaptk160_true_txl_warmstart
      --source-checkpoint "${SOURCE_CHECKPOINT}"
      --target-checkpoint "${TARGET_CHECKPOINT}"
      --output-json "${OUTPUT_JSON}"
      --device "cuda:${GPU_ID}"
    )
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf 'cd %q && ' "${ROOT}"
      printf 'PYTHONPATH=%q ' "${PROJECT_ROOT}/src"
      printf '%q ' "${cmd[@]}"
      printf '\n'
      exit 0
    fi
    mkdir -p "$(dirname "${TARGET_CHECKPOINT}")" "$(dirname "${OUTPUT_JSON}")"
    cd "${ROOT}"
    export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export WANDB_DISABLED=true
    exec "${cmd[@]}"
    ;;
  *)
    echo "unknown stage: ${STAGE}" >&2
    usage >&2
    exit 2
    ;;
esac

NUM_ENVS="${NUM_ENVS:-${DEFAULT_NUM_ENVS}}"
MAX_ITER="${MAX_ITER:-${DEFAULT_MAX_ITER}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${DEFAULT_SAVE_INTERVAL}}"
SEED="${SEED:-${DEFAULT_SEED}}"
LEARNING_RATE="${LEARNING_RATE:-${DEFAULT_LEARNING_RATE}}"
ENTROPY_COEF="${ENTROPY_COEF:-${DEFAULT_ENTROPY_COEF}}"
EXPERIMENT="${EXPERIMENT:-task048_previous_gait_${STAGE}}"
RUN_NAME="${RUN_NAME:-048_${STAGE}_${PROFILE}_env${NUM_ENVS}_iter${MAX_ITER}_seed${SEED}}"

resume_args=()
if [[ "${STAGE}" != "mlp-prior" ]]; then
  if [[ -z "${SOURCE_CHECKPOINT:-}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "${STAGE} requires an existing SOURCE_CHECKPOINT" >&2
    exit 3
  fi
  SOURCE_CHECKPOINT="$(realpath "${SOURCE_CHECKPOINT}")"
  LOAD_RUN="${LOAD_RUN:-task048_${STAGE}_warmstart_source}"
  LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-$(basename "${SOURCE_CHECKPOINT}")}"
  resume_args=(
    --agent.resume=True
    "--agent.load-run=${LOAD_RUN}"
    "--agent.load-checkpoint=${LOAD_CHECKPOINT}"
  )
fi

cmd=(
  "${PY}" scripts/train.py "${TASK}"
  "--gpu-ids=[${GPU_ID}]"
  "--env.scene.num-envs=${NUM_ENVS}"
  "--agent.max-iterations=${MAX_ITER}"
  "--agent.save-interval=${SAVE_INTERVAL}"
  "--agent.experiment-name=${EXPERIMENT}"
  "--agent.run-name=${RUN_NAME}"
  --agent.logger=tensorboard
  --agent.upload-model=False
  "--agent.seed=${SEED}"
  "--agent.algorithm.learning-rate=${LEARNING_RATE}"
  "--agent.algorithm.entropy-coef=${ENTROPY_COEF}"
  --agent.algorithm.num-learning-epochs=5
  --agent.algorithm.num-mini-batches=4
  "${resume_args[@]}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'cd %q && ' "${ROOT}"
  printf 'PYTHONPATH=%q ' "${PROJECT_ROOT}/src"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUT}"
if [[ "${STAGE}" != "mlp-prior" ]]; then
  mkdir -p "${ROOT}/logs/rsl_rl/${EXPERIMENT}/${LOAD_RUN}"
  ln -sfn "${SOURCE_CHECKPOINT}" \
    "${ROOT}/logs/rsl_rl/${EXPERIMENT}/${LOAD_RUN}/${LOAD_CHECKPOINT}"
fi

cd "${ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_DISABLED=true

set +e
"${cmd[@]}" 2>&1 | tee "${OUT}/${RUN_NAME}.stdout.log"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
