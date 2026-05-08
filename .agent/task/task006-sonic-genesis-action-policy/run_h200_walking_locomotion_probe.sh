#!/usr/bin/env bash
set -euo pipefail

REPO=/root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke
TASK=/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy
UPSTREAM=/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl

cd /tmp
args=(
  --asset "$UPSTREAM/gear_sonic/data/robots/g1/g1_29dof.xml" \
  --decoder "$UPSTREAM/gear_sonic_deploy/policy/release/model_decoder.onnx" \
  --obs-csv "$TASK/actions/official_policy_input_walking_capture.csv" \
  --frames "${FRAMES:-100}" \
  --token-mode replay \
  --history-init official_obs \
  --root-qpos 0.002389 0.011728 0.791166 0.999910712 -0.006119614 0.011878765 0.000052081 \
  --initial-joint-pos-csv "$TASK/actions/official_walking_q_log_300f.csv" \
  --initial-joint-pos-row 0 \
  --log-every "${LOG_EVERY:-20}" \
  --min-horizontal-displacement "${MIN_HORIZONTAL_DISPLACEMENT:-0.05}" \
  --height-ok-min 0.3 \
  --height-ok-max 1.2
)

if [[ "${HEARTBEAT_EVERY_FRAME:-0}" == "1" ]]; then
  args+=(--heartbeat-every-frame)
fi

if [[ -n "${PROGRESS_FILE:-}" ]]; then
  args+=(--progress-file "$PROGRESS_FILE")
fi

if [[ "${SKIP_FOOT_METRICS:-0}" == "1" ]]; then
  args+=(--skip-foot-metrics)
fi

if [[ -n "${MAX_WALL_TIME_S:-}" ]]; then
  args+=(--max-wall-time-s "$MAX_WALL_TIME_S")
fi

PYTHONPATH="$REPO/src" python3 -u -m h200_locomotion_lab.tools.genesis_sonic_policy_locomotion_probe "${args[@]}"
