#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_ROOT="/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl"
DEPLOY_DIR="${UPSTREAM_ROOT}/gear_sonic_deploy"
TASK_ROOT="/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy"
TRT_ROOT="/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/trt-10.13.3-cuda12.9-root/usr"
WALKING_NAME="walking_quip_360_R_002__A428"
WALKING_SRC="${DEPLOY_DIR}/reference/example/${WALKING_NAME}"
WALKING_REF_ROOT="${TASK_ROOT}/reference_walking_only"

mkdir -p "${TASK_ROOT}/logs" "${TASK_ROOT}/actions" "${TASK_ROOT}/official_deploy_logs_walking" "${WALKING_REF_ROOT}"
ln -sfn "${WALKING_SRC}" "${WALKING_REF_ROOT}/${WALKING_NAME}"

SIM_LOG="${TASK_ROOT}/logs/run_sim_loop_walking_capture.log"
if ! pgrep -f "gear_sonic/scripts/run_sim_loop.py" >/dev/null; then
  (
    cd "${UPSTREAM_ROOT}"
    source .venv_sim/bin/activate
    exec xvfb-run -a python gear_sonic/scripts/run_sim_loop.py
  ) >"${SIM_LOG}" 2>&1 &
  echo "$!" >"${TASK_ROOT}/logs/run_sim_loop_walking_capture.pid"
  sleep 8
fi

cd "${DEPLOY_DIR}"
set +e
set +u
export TensorRT_ROOT=/root/TensorRT
source scripts/setup_env.sh >/dev/null 2>&1
set -e
set -u
export TensorRT_ROOT="${TRT_ROOT}"
export LD_LIBRARY_PATH="${TRT_ROOT}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

DEPLOY_LOG="${TASK_ROOT}/logs/official_deploy_walking_capture.log"
POLICY_INPUT="${TASK_ROOT}/actions/official_policy_input_walking_capture.csv"
TARGET_MOTION="${TASK_ROOT}/actions/official_target_motion_walking_capture.csv"
PLANNER_MOTION="${TASK_ROOT}/actions/official_planner_motion_walking_capture.csv"
STDIN_FIFO="${TASK_ROOT}/logs/official_deploy_walking_stdin.fifo"
rm -f "${STDIN_FIFO}"
mkfifo "${STDIN_FIFO}"

tail -f /dev/null >"${STDIN_FIFO}" &
FIFO_HOLDER=$!

./target/release/g1_deploy_onnx_ref \
  lo \
  policy/release/model_decoder.onnx \
  "${WALKING_REF_ROOT}" \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type manager \
  --output-type all \
  --zmq-host localhost \
  --disable-crc-check \
  --policy-input-logfile "${POLICY_INPUT}" \
  --target-motion-logfile "${TARGET_MOTION}" \
  --planner-motion-logfile "${PLANNER_MOTION}" \
  --logs-dir "${TASK_ROOT}/official_deploy_logs_walking" \
  --enable-csv-logs \
  <"${STDIN_FIFO}" >"${DEPLOY_LOG}" 2>&1 &
DEPLOY_PID=$!

ready=0
for _ in $(seq 1 180); do
  if grep -q "G1Deploy object created successfully" "${DEPLOY_LOG}"; then
    ready=1
    break
  fi
  if ! kill -0 "${DEPLOY_PID}" 2>/dev/null; then
    break
  fi
  sleep 1
done
echo "[HARNESS] ready=${ready} at $(date -Iseconds)" >>"${DEPLOY_LOG}"
if [ "${ready}" != "1" ]; then
  kill "${DEPLOY_PID}" 2>/dev/null || true
  wait "${DEPLOY_PID}" || true
  kill "${FIFO_HOLDER}" 2>/dev/null || true
  rm -f "${STDIN_FIFO}"
  exit 1
fi

printf ']' >"${STDIN_FIFO}"
echo "[HARNESS] sent_start_right_bracket at $(date -Iseconds)" >>"${DEPLOY_LOG}"
sleep 5
printf 'T' >"${STDIN_FIFO}"
echo "[HARNESS] sent_play_T at $(date -Iseconds)" >>"${DEPLOY_LOG}"
sleep 16
printf 'O' >"${STDIN_FIFO}"
echo "[HARNESS] sent_stop_O at $(date -Iseconds)" >>"${DEPLOY_LOG}"

set +e
wait "${DEPLOY_PID}"
DEPLOY_STATUS=$?
set -e
kill "${FIFO_HOLDER}" 2>/dev/null || true
rm -f "${STDIN_FIFO}"
echo "[HARNESS] deploy_status=${DEPLOY_STATUS}" >>"${DEPLOY_LOG}"

python3 - <<PY
from pathlib import Path
import csv, math
policy = Path("${POLICY_INPUT}")
target = Path("${TARGET_MOTION}")
print("DEPLOY_STATUS", ${DEPLOY_STATUS})
print("DEPLOY_LOG", "${DEPLOY_LOG}")
print("POLICY_INPUT", policy)
print("TARGET_MOTION", target)
if policy.exists():
    rows = [tuple(float(x) for x in row if x.strip()) for row in csv.reader(policy.open()) if row]
    dims = sorted({len(row) for row in rows})
    print("POLICY_ROWS", len(rows))
    print("POLICY_DIMS", dims)
    print("POLICY_FINITE", all(math.isfinite(x) for row in rows for x in row))
if target.exists():
    rows = [tuple(float(x) for x in row if x.strip()) for row in csv.reader(target.open()) if row]
    dims = sorted({len(row) for row in rows})
    print("TARGET_ROWS", len(rows))
    print("TARGET_DIMS", dims)
    if rows and len(rows[0]) >= 3:
        dx = rows[-1][0] - rows[0][0]
        dy = rows[-1][1] - rows[0][1]
        path = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(rows, rows[1:]))
        print("TARGET_START_XYZ", rows[0][:3])
        print("TARGET_FINAL_XYZ", rows[-1][:3])
        print("TARGET_DISP_XY", math.hypot(dx, dy))
        print("TARGET_PATH_XY", path)
PY

exit "${DEPLOY_STATUS}"
