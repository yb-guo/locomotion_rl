#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter}"
MJLAB_ROOT="${MJLAB_ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab}"
PY="${PY:-/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python}"
CHECKPOINT="${CHECKPOINT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task035_eval_gated_curriculum_train/2026-05-28_15-31-23_035_mixed_from_model5350_env8192_iter20_gpu1_seed3503601/model_5369.pt}"
LABEL="${LABEL:-model_5369}"
OUT="${OUT:-${ROOT}/outputs/task035/full_validation_${LABEL}}"
DEVICE="${DEVICE:-cuda:0}"
DYNAMIC_TASK="${DYNAMIC_TASK:-Unitree-G1-Gripper-Flat-Task033-StackMlpK4-DynamicMotorFailure-Fast1p6}"
DEADGRID_TASK="${DEADGRID_TASK:-Unitree-G1-Gripper-Flat-Task033-StackMlpK4-FocusedDeadGrid-Fast2p0}"
SEED="${SEED:-3503800}"
STEPS="${STEPS:-500}"
DYNAMIC_NUM_ENVS="${DYNAMIC_NUM_ENVS:-256}"
DEADGRID_NUM_ENVS="${DEADGRID_NUM_ENVS:-128}"
SPEEDS="${SPEEDS:-0.4 1.2 2.0}"

mkdir -p "${OUT}"
cd "${ROOT}"

export PYTHONPATH="/tmp/task029_ipython_stub:/tmp/task029_pydeps:${ROOT}/src:${MJLAB_ROOT}:${MJLAB_ROOT}/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OUT LABEL CHECKPOINT

for vx in ${SPEEDS}; do
  vx_tag="${vx/./p}"
  case_dir="${OUT}/vx${vx_tag}"
  mkdir -p "${case_dir}/dynamic_switch" "${case_dir}/deadgrid"
  "${PY}" -m h200_locomotion_lab.tools.task033_dynamic_eval_checkpoint \
    --task "${DYNAMIC_TASK}" --checkpoint "${CHECKPOINT}" \
    --output-json "${case_dir}/dynamic_switch/task033_dynamic_eval_switch_vx${vx_tag}.json" \
    --lin-vel-x "${vx}" --num-envs "${DYNAMIC_NUM_ENVS}" \
    --steps "${STEPS}" --seed "${SEED}" --device "${DEVICE}"
  "${PY}" -m h200_locomotion_lab.tools.task033_failure_grid_eval_checkpoint \
    --task "${DEADGRID_TASK}" --checkpoint "${CHECKPOINT}" \
    --output-dir "${case_dir}/deadgrid" --lin-vel-x "${vx}" \
    --num-envs "${DEADGRID_NUM_ENVS}" --steps "${STEPS}" \
    --seed "${SEED}" --device "${DEVICE}"
done

"${PY}" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"]).resolve()
records = []
for vx_dir in sorted(path for path in out.iterdir() if path.is_dir() and path.name.startswith("vx")):
    speed = float(vx_dir.name[2:].replace("p", "."))
    dynamic_paths = list((vx_dir / "dynamic_switch").glob("*.json"))
    deadgrid_path = vx_dir / "deadgrid" / "task033_failure_grid_eval_aggregate.json"
    item = {"speed": speed, "speed_tag": vx_dir.name}
    if dynamic_paths:
        data = json.loads(dynamic_paths[0].read_text())
        item["dynamic_switch"] = {
            "pass": data.get("pass"),
            "zero_fall_ratio": data.get("zero_fall_ratio"),
            "recovery_success_ratio": data.get("recovery_success_ratio"),
            "path": str(dynamic_paths[0]),
        }
    if deadgrid_path.exists():
        data = json.loads(deadgrid_path.read_text())
        item["deadgrid"] = {
            "pass": data.get("pass"),
            "pass_count": data.get("pass_count"),
            "case_count": data.get("grid_case_count"),
            "failed": data.get("failed", []),
            "path": str(deadgrid_path),
        }
    item["pass"] = item.get("dynamic_switch", {}).get("pass") is True and item.get("deadgrid", {}).get("pass") is True
    records.append(item)

summary = {
    "task": "task035-eval-gated-history-curriculum",
    "phase": "full_validation",
    "checkpoint": os.environ["CHECKPOINT"],
    "label": os.environ["LABEL"],
    "output_dir": str(out),
    "record_count": len(records),
    "pass": bool(records) and all(item.get("pass") for item in records),
    "records": records,
    "failed": [item for item in records if not item.get("pass")],
}
summary_path = out / "task035_full_validation_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
