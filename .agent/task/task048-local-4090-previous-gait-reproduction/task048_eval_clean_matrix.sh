#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -ne 2 ]]; then
  cat <<'USAGE'
usage: bash task048_eval_clean_matrix.sh ARCH CHECKPOINT

ARCH is one of: mlp, adaptk4, adaptk160, true-txl.

Environment overrides:
  ROOT, PROJECT_ROOT, PY, GPU_ID, NUM_ENVS, STEPS, TRIAL_LENGTH_S, SEED,
  OUTPUT_DIR.
USAGE
  [[ $# -eq 2 ]] && exit 0
  exit 2
fi

ARCH="$1"
CHECKPOINT="$2"
ROOT="${ROOT:-/home/xyzl/yubo/locomotion_rl/external/unitree_rl_mjlab}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/xyzl/yubo/locomotion_rl}"
PY="${PY:-${PROJECT_ROOT}/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
NUM_ENVS="${NUM_ENVS:-64}"
STEPS="${STEPS:-360}"
TRIAL_LENGTH_S="${TRIAL_LENGTH_S:-2.0}"
SEED="${SEED:-4800301}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/task048/eval/${ARCH}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "missing checkpoint: ${CHECKPOINT}" >&2
  exit 3
fi
CHECKPOINT="$(realpath "${CHECKPOINT}")"

extra_actor_args=()
case "${ARCH}" in
  mlp)
    TASK="Unitree-G1-Gripper-Flat-Task048-Mlp-CleanBins-Eval"
    ;;
  adaptk4)
    TASK="Unitree-G1-Gripper-Flat-Task048-AdaptK4-CleanBins-Eval"
    ;;
  adaptk160)
    TASK="Unitree-G1-Gripper-Flat-Task048-AdaptK160-CleanBins"
    ;;
  true-txl)
    TASK="Unitree-G1-Gripper-Flat-Task048-TrueTxl-CleanBins-Eval"
    extra_actor_args=(
      --memory-latent-dim 32
      --base-obs-passthrough
      --adaptation-warmstart
      --action-dim 31
      --adaptation-hidden-dim 128
    )
    ;;
  *)
    echo "unknown architecture: ${ARCH}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
case_files=()
speeds=(0.4 1.2 2.0)
labels=(0p4 1p2 2p0)
max_lin_errors=(0.25 0.55 0.90)

cd "${ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_DISABLED=true

for index in "${!speeds[@]}"; do
  speed="${speeds[$index]}"
  label="${labels[$index]}"
  max_lin_error="${max_lin_errors[$index]}"
  output_json="${OUTPUT_DIR}/${ARCH}_vx${label}.json"
  case_files+=("${output_json}")
  "${PY}" -m h200_locomotion_lab.tools.task037_multitrial_eval_checkpoint \
    --task "${TASK}" \
    --checkpoint "${CHECKPOINT}" \
    --output-json "${output_json}" \
    --num-envs "${NUM_ENVS}" \
    --steps "${STEPS}" \
    --trial-length-s "${TRIAL_LENGTH_S}" \
    --lin-vel-x "${speed}" \
    --lin-vel-y 0.0 \
    --ang-vel-z 0.0 \
    --seed "$((SEED + index))" \
    --device "cuda:${GPU_ID}" \
    --min-final-completion-ratio 0.95 \
    --max-final-fall-ratio 0.0 \
    --max-final-lin-vel-error "${max_lin_error}" \
    --max-final-yaw-vel-error 0.35 \
    --max-final-gravity-xy 0.75 \
    --min-final-root-z 0.55 \
    "${extra_actor_args[@]}"
done

summary_json="${OUTPUT_DIR}/${ARCH}_clean_matrix_summary.json"
"${PY}" - "${summary_json}" "${ARCH}" "${CHECKPOINT}" "${case_files[@]}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
architecture = sys.argv[2]
checkpoint = sys.argv[3]
case_paths = [Path(value) for value in sys.argv[4:]]
cases = [json.loads(path.read_text(encoding="utf-8")) for path in case_paths]
case_passes = [
    bool(case.get("pass"))
    and float((case.get("final_trial") or {}).get("fall_ratio", 1.0)) == 0.0
    for case in cases
]
matrix_pass = len(cases) == 3 and all(case_passes)
summary = {
    "schema": "task048_clean_gait_matrix_v1",
    "architecture": architecture,
    "checkpoint": checkpoint,
    "speeds_mps": [0.4, 1.2, 2.0],
    "case_jsons": [str(path) for path in case_paths],
    "cases": cases,
    "case_passes": case_passes,
    "zero_fall_required": True,
    "matrix_pass": matrix_pass,
    "pass": matrix_pass,
    "quality_claim": False,
    "training_claim": False,
    "eval_claim": False,
    "reproduction_claim": False,
    "superiority_claim": False,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
raise SystemExit(0 if matrix_pass else 1)
PY
