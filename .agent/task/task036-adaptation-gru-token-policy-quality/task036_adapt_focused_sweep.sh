#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter}"
MJLAB_ROOT="${MJLAB_ROOT:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab}"
PY="${PY:-/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python}"
TASK="${TASK:-Unitree-G1-Gripper-Flat-Task036-AdaptK4-FocusedDeadGrid-Fast2p0}"
DEVICE="${DEVICE:-cuda:0}"
RUN_DIR="${RUN_DIR:-/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task036_policy_quality_train/2026-05-28_23-10-53_036_adapt_k4_warmstart_env8192_iter60_gpu0_seed3603630}"
OUT="${OUT:-${ROOT}/outputs/task036/adapt_focused_sweep}"
SEED="${SEED:-3603900}"
STEPS="${STEPS:-500}"
NUM_ENVS="${NUM_ENVS:-128}"
CHECKPOINT_IDS="${CHECKPOINT_IDS:-5398 5400 5402 5404 5406 5408}"

export PYTHONPATH="/tmp/task029_ipython_stub:/tmp/task029_pydeps:${ROOT}/src:${MJLAB_ROOT}:${MJLAB_ROOT}/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "${OUT}"
export OUT

if [[ "${SUMMARY_ONLY:-0}" != "1" ]]; then
  for checkpoint_id in ${CHECKPOINT_IDS}; do
    checkpoint="${RUN_DIR}/model_${checkpoint_id}.pt"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "missing checkpoint: ${checkpoint}" >&2
      continue
    fi
    for speed in 0.4 2.0; do
      speed_tag="${speed/./p}"
      case "${speed}" in
        0.4)
          joints=(left_hip_yaw_joint left_hip_roll_joint right_knee_joint)
          ;;
        2.0)
          joints=(right_knee_joint)
          ;;
        *)
          echo "unexpected speed ${speed}" >&2
          exit 2
          ;;
      esac
      output_dir="${OUT}/model_${checkpoint_id}/vx${speed_tag}"
      cmd=(
        "${PY}" -m h200_locomotion_lab.tools.task033_failure_grid_eval_checkpoint
        --task "${TASK}"
        --checkpoint "${checkpoint}"
        --output-dir "${output_dir}"
        --lin-vel-x "${speed}"
        --num-envs "${NUM_ENVS}"
        --steps "${STEPS}"
        --seed "${SEED}"
        --device "${DEVICE}"
        --joints "${joints[@]}"
      )
      "${cmd[@]}"
    done
  done
fi

python3 - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ.get("OUT", "")).resolve()
records = []
for model_dir in sorted(out.glob("model_*")):
    checkpoint_id = model_dir.name.removeprefix("model_")
    item = {"checkpoint_id": checkpoint_id, "model_dir": str(model_dir), "speeds": []}
    for speed_dir in sorted(model_dir.glob("vx*")):
        aggregate_path = speed_dir / "task033_failure_grid_eval_aggregate.json"
        if not aggregate_path.exists():
            item["speeds"].append({"speed_tag": speed_dir.name, "pass": False, "missing": True})
            continue
        data = json.loads(aggregate_path.read_text())
        item["speeds"].append(
            {
                "speed_tag": speed_dir.name,
                "pass": data.get("pass"),
                "pass_count": data.get("pass_count"),
                "case_count": data.get("grid_case_count"),
                "failed_joints": [case.get("joint_name") for case in data.get("failed", [])],
                "path": str(aggregate_path),
            }
        )
    item["pass"] = bool(item["speeds"]) and all(speed.get("pass") for speed in item["speeds"])
    records.append(item)

summary = {
    "task": "task036-adaptation-gru-token-policy-quality",
    "phase": "adapt_focused_checkpoint_sweep",
    "output_dir": str(out),
    "pass": any(item.get("pass") for item in records),
    "records": records,
    "passing_checkpoint_ids": [item["checkpoint_id"] for item in records if item.get("pass")],
}
summary_path = out / "task036_adapt_focused_sweep_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
