#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter}"
MJLAB_ROOT="${MJLAB_ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab}"
PY="${PY:-/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python}"
CHECKPOINT="${CHECKPOINT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_stackmlp_k4_frozenbase_focused/2026-05-28_12-40-56_033_frozenbase_focused_from5349_env8192_iter30_gpu1_seed3303362_lr1e5/model_5350.pt}"
OUT="${OUT:-${ROOT}/outputs/task035/model5350_baseline_gate}"
DEVICE="${DEVICE:-cuda:0}"
DYNAMIC_TASK="${DYNAMIC_TASK:-Unitree-G1-Gripper-Flat-Task033-StackMlpK4-DynamicMotorFailure-Fast1p6}"
DEADGRID_TASK="${DEADGRID_TASK:-Unitree-G1-Gripper-Flat-Task033-StackMlpK4-FocusedDeadGrid-Fast2p0}"
DYNAMIC_NUM_ENVS="${DYNAMIC_NUM_ENVS:-256}"
DEADGRID_NUM_ENVS="${DEADGRID_NUM_ENVS:-128}"
STEPS="${STEPS:-500}"
RUN_DYNAMIC="${RUN_DYNAMIC:-1}"
RUN_DEADGRID="${RUN_DEADGRID:-1}"

mkdir -p "${OUT}"
cd "${ROOT}"

IPYTHON_STUB="${OUT}/_ipython_stub"
TASK029_HEADLESS_STUB="${TASK029_HEADLESS_STUB:-/tmp/task029_ipython_stub}"
mkdir -p "${IPYTHON_STUB}/IPython"
cat > "${IPYTHON_STUB}/IPython/__init__.py" <<'PY'
__version__ = "task035-headless-stub"
PY
cat > "${IPYTHON_STUB}/IPython/display.py" <<'PY'
class HTML:
    def __init__(self, data=None, *args, **kwargs):
        self.data = data

class Image:
    def __init__(self, data=None, *args, **kwargs):
        self.data = data

class Video:
    def __init__(self, data=None, *args, **kwargs):
        self.data = data

def display(*args, **kwargs):
    return None
PY

if [[ -d "${TASK029_HEADLESS_STUB}" ]]; then
  export PYTHONPATH="${TASK029_HEADLESS_STUB}:${IPYTHON_STUB}:${ROOT}/src:${MJLAB_ROOT}:${MJLAB_ROOT}/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="${IPYTHON_STUB}:${ROOT}/src:${MJLAB_ROOT}:${MJLAB_ROOT}/src:${PYTHONPATH:-}"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OUT

seeds=(3503500 3503501 3503502)

for seed in "${seeds[@]}"; do
  case_dir="${OUT}/vx2p0/seed${seed}"
  mkdir -p "${case_dir}/dynamic_switch" "${case_dir}/deadgrid"

  if [[ "${RUN_DYNAMIC}" == "1" ]]; then
    "${PY}" -m h200_locomotion_lab.tools.task033_dynamic_eval_checkpoint \
      --task "${DYNAMIC_TASK}" \
      --checkpoint "${CHECKPOINT}" \
      --output-json "${case_dir}/dynamic_switch/task033_dynamic_eval_switch_vx2p0.json" \
      --lin-vel-x 2.0 \
      --num-envs "${DYNAMIC_NUM_ENVS}" \
      --steps "${STEPS}" \
      --seed "${seed}" \
      --device "${DEVICE}"
  fi

  if [[ "${RUN_DEADGRID}" == "1" ]]; then
    "${PY}" -m h200_locomotion_lab.tools.task033_failure_grid_eval_checkpoint \
      --task "${DEADGRID_TASK}" \
      --checkpoint "${CHECKPOINT}" \
      --output-dir "${case_dir}/deadgrid" \
      --lin-vel-x 2.0 \
      --num-envs "${DEADGRID_NUM_ENVS}" \
      --steps "${STEPS}" \
      --seed "${seed}" \
      --device "${DEVICE}"
  fi
done

"${PY}" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"]).resolve()
records = []
for path in sorted(out.glob("vx2p0/seed*/dynamic_switch/*.json")):
    data = json.loads(path.read_text())
    records.append({
        "case": "dynamic_switch",
        "path": str(path),
        "pass": data.get("pass"),
        "speed": data.get("fixed_lin_vel_x"),
        "seed": data.get("seed"),
        "zero_fall_ratio": data.get("zero_fall_ratio"),
        "recovery_success_ratio": data.get("recovery_success_ratio"),
    })
for path in sorted(out.glob("vx2p0/seed*/deadgrid/task033_failure_grid_eval_aggregate.json")):
    data = json.loads(path.read_text())
    records.append({
        "case": "deadgrid",
        "path": str(path),
        "pass": data.get("pass"),
        "speed": data.get("fixed_command", {}).get("lin_vel_x"),
        "seed": data.get("seed"),
        "pass_count": data.get("pass_count"),
        "case_count": data.get("grid_case_count"),
        "failed": data.get("failed", []),
    })
summary = {
    "task": "task035-eval-gated-history-curriculum",
    "phase": "model5350_baseline_gate",
    "output_dir": str(out),
    "record_count": len(records),
    "pass": bool(records) and all(item.get("pass") for item in records),
    "records": records,
    "failed": [item for item in records if not item.get("pass")],
}
summary_path = out / "task035_model5350_baseline_gate_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
