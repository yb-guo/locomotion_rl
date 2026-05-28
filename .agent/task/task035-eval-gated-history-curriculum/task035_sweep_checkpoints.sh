#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter}"
MJLAB_ROOT="${MJLAB_ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab}"
PY="${PY:-/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python}"
RUN_DIR="${RUN_DIR:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task035_eval_gated_curriculum_train/2026-05-28_15-31-23_035_mixed_from_model5350_env8192_iter20_gpu1_seed3503601}"
OUT="${OUT:-${ROOT}/outputs/task035/checkpoint_sweep_mixed_seed3503601}"
DEVICE="${DEVICE:-cuda:0}"
DYNAMIC_TASK="${DYNAMIC_TASK:-Unitree-G1-Gripper-Flat-Task033-StackMlpK4-DynamicMotorFailure-Fast1p6}"
DEADGRID_TASK="${DEADGRID_TASK:-Unitree-G1-Gripper-Flat-Task033-StackMlpK4-FocusedDeadGrid-Fast2p0}"
SEED="${SEED:-3503700}"
STEPS="${STEPS:-500}"
DYNAMIC_NUM_ENVS="${DYNAMIC_NUM_ENVS:-256}"
DEADGRID_NUM_ENVS="${DEADGRID_NUM_ENVS:-128}"
CHECKPOINTS="${CHECKPOINTS:-${RUN_DIR}/model_5352.pt ${RUN_DIR}/model_5360.pt ${RUN_DIR}/model_5369.pt}"

mkdir -p "${OUT}"
cd "${ROOT}"

export PYTHONPATH="/tmp/task029_ipython_stub:/tmp/task029_pydeps:${ROOT}/src:${MJLAB_ROOT}:${MJLAB_ROOT}/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OUT

for checkpoint in ${CHECKPOINTS}; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 3
  fi
  label="$(basename "${checkpoint}" .pt)"
  case_dir="${OUT}/${label}"
  mkdir -p "${case_dir}/dynamic_switch" "${case_dir}/right_knee_dead"

  "${PY}" -m h200_locomotion_lab.tools.task033_dynamic_eval_checkpoint \
    --task "${DYNAMIC_TASK}" \
    --checkpoint "${checkpoint}" \
    --output-json "${case_dir}/dynamic_switch/task033_dynamic_eval_switch_vx2p0.json" \
    --lin-vel-x 2.0 \
    --num-envs "${DYNAMIC_NUM_ENVS}" \
    --steps "${STEPS}" \
    --seed "${SEED}" \
    --device "${DEVICE}"

  "${PY}" -m h200_locomotion_lab.tools.task033_failure_grid_eval_checkpoint \
    --task "${DEADGRID_TASK}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${case_dir}/right_knee_dead" \
    --joints right_knee_joint \
    --lin-vel-x 2.0 \
    --num-envs "${DEADGRID_NUM_ENVS}" \
    --steps "${STEPS}" \
    --seed "${SEED}" \
    --device "${DEVICE}"
done

"${PY}" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"]).resolve()
records = []
for model_dir in sorted(path for path in out.iterdir() if path.is_dir()):
    dynamic_path = model_dir / "dynamic_switch" / "task033_dynamic_eval_switch_vx2p0.json"
    right_knee_path = model_dir / "right_knee_dead" / "task033_failure_grid_eval_aggregate.json"
    item = {"checkpoint_label": model_dir.name}
    if dynamic_path.exists():
        data = json.loads(dynamic_path.read_text())
        item["dynamic_switch"] = {
            "pass": data.get("pass"),
            "zero_fall_ratio": data.get("zero_fall_ratio"),
            "recovery_success_ratio": data.get("recovery_success_ratio"),
            "path": str(dynamic_path),
        }
    if right_knee_path.exists():
        data = json.loads(right_knee_path.read_text())
        item["right_knee_dead"] = {
            "pass": data.get("pass"),
            "pass_count": data.get("pass_count"),
            "case_count": data.get("grid_case_count"),
            "failed": data.get("failed", []),
            "path": str(right_knee_path),
        }
    item["pass"] = (
        item.get("dynamic_switch", {}).get("pass") is True
        and item.get("right_knee_dead", {}).get("pass") is True
    )
    records.append(item)

summary = {
    "task": "task035-eval-gated-history-curriculum",
    "phase": "checkpoint_sweep_fast_gate",
    "output_dir": str(out),
    "record_count": len(records),
    "pass_count": sum(1 for item in records if item.get("pass")),
    "records": records,
}
summary["best_candidates"] = [item for item in records if item.get("pass")]
summary_path = out / "task035_checkpoint_sweep_fast_gate_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
