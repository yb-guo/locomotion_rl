#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter}"
MJLAB_ROOT="${MJLAB_ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab}"
PY="${PY:-/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
LABEL="${LABEL:-$(basename "${CHECKPOINT}" .pt)}"
OUT="${OUT:-${ROOT}/outputs/task037/full_validation_${LABEL}}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-3700800}"
STEPS="${STEPS:-360}"
NUM_ENVS="${NUM_ENVS:-64}"
SPEEDS="${SPEEDS:-0.4 1.2 2.0}"
DYNAMIC_TASK="${DYNAMIC_TASK:-Unitree-G1-Gripper-Flat-Task037-TxlMemoryK160-DynamicMotorFailure-Fast1p6}"
DEADGRID_TASK="${DEADGRID_TASK:-Unitree-G1-Gripper-Flat-Task037-TxlMemoryK160-FocusedDeadGrid-Fast2p0}"
TRIAL_LENGTH_S="${TRIAL_LENGTH_S:-2.0}"
JOINTS="${JOINTS:-left_hip_pitch_joint left_hip_yaw_joint right_hip_pitch_joint right_hip_yaw_joint left_hip_roll_joint left_knee_joint right_hip_roll_joint right_knee_joint left_ankle_pitch_joint left_ankle_roll_joint right_ankle_pitch_joint right_ankle_roll_joint}"

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
  "${PY}" -m h200_locomotion_lab.tools.task037_multitrial_eval_checkpoint \
    --task "${DYNAMIC_TASK}" --checkpoint "${CHECKPOINT}" \
    --output-json "${case_dir}/dynamic_switch/task037_multitrial_dynamic_switch_vx${vx_tag}.json" \
    --dynamic-case switch --lin-vel-x "${vx}" --num-envs "${NUM_ENVS}" \
    --steps "${STEPS}" --trial-length-s "${TRIAL_LENGTH_S}" \
    --seed "${SEED}" --device "${DEVICE}"

  idx=0
  for joint in ${JOINTS}; do
    joint_dir="${case_dir}/deadgrid/${idx}_${joint}"
    mkdir -p "${joint_dir}"
    "${PY}" -m h200_locomotion_lab.tools.task037_multitrial_eval_checkpoint \
      --task "${DEADGRID_TASK}" --checkpoint "${CHECKPOINT}" \
      --output-json "${joint_dir}/task037_multitrial_dead_${joint}_vx${vx_tag}.json" \
      --force-dead-joint "${joint}" --lin-vel-x "${vx}" --num-envs "${NUM_ENVS}" \
      --steps "${STEPS}" --trial-length-s "${TRIAL_LENGTH_S}" \
      --seed "$((SEED + idx + 1))" --device "${DEVICE}"
    idx=$((idx + 1))
  done
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
    item = {"speed": speed, "speed_tag": vx_dir.name}
    if dynamic_paths:
        data = json.loads(dynamic_paths[0].read_text())
        item["dynamic_switch"] = {
            "pass": data.get("pass"),
            "final_trial_pass": data.get("final_trial_pass"),
            "final_trial_fall_ratio": data.get("final_trial", {}).get("fall_ratio"),
            "final_trial_lin_vel_error_mean": data.get("final_trial", {}).get("lin_vel_error", {}).get("mean"),
            "path": str(dynamic_paths[0]),
        }
    dead_records = []
    for path in sorted((vx_dir / "deadgrid").glob("*/*.json")):
        data = json.loads(path.read_text())
        dead_records.append(
            {
                "joint": data.get("force_dead_joint"),
                "pass": data.get("pass"),
                "final_trial_pass": data.get("final_trial_pass"),
                "final_trial_fall_ratio": data.get("final_trial", {}).get("fall_ratio"),
                "final_trial_lin_vel_error_mean": data.get("final_trial", {}).get("lin_vel_error", {}).get("mean"),
                "path": str(path),
            }
        )
    item["deadgrid"] = {
        "case_count": len(dead_records),
        "pass_count": sum(1 for record in dead_records if record.get("pass") is True),
        "pass": bool(dead_records) and all(record.get("pass") is True for record in dead_records),
        "failed": [record for record in dead_records if record.get("pass") is not True],
    }
    item["pass"] = item.get("dynamic_switch", {}).get("pass") is True and item["deadgrid"]["pass"] is True
    records.append(item)

summary = {
    "task": "task037-locoformer-style-multitrial-long-context-training",
    "phase": "full_validation",
    "checkpoint": os.environ["CHECKPOINT"],
    "label": os.environ["LABEL"],
    "output_dir": str(out),
    "record_count": len(records),
    "promotion_gate": "final_trial_pass",
    "pass": bool(records) and all(item.get("pass") for item in records),
    "records": records,
    "failed": [item for item in records if not item.get("pass")],
}
summary_path = out / "task037_full_validation_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
