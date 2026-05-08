# Route

Task: task007-sonic-g1-deployment-bridge

Goal: run the backend dry-run tool on the H200 target using existing task006
official/Genesis logs.

Pass condition:

- H200 dry-run command completes.
- Summary CSV is written.
- Observation/action/target finite gates pass.
- Result is recorded in `run_h200_notes.md`.

# Log

- 2026-05-08: Synced task007 backend dry-run files to the H200 repo copy:

```text
/root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke
```

Targeted H200 tests from `/tmp` with explicit `PYTHONPATH`:

```text
PYTHONPATH=/root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke/src \
python3 -m pytest -p no:cacheprovider \
  tests/test_robot_backend.py \
  tests/test_sonic_g1_deployment_dry_run.py \
  tests/test_genesis_sonic_planner_encoder_rollout_probe.py

10 passed
```

H200 dry-run command class:

```text
python3 -m h200_locomotion_lab.tools.sonic_g1_deployment_dry_run \
  --qpos-csv /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/planner_ort_walk_qpos.csv \
  --raw-action-csv /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_walking_raw_actions_log_300f.csv \
  --frames 64 \
  --output-summary-csv /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/sonic_g1_deployment_dry_run_planner_qpos_official_actions_64f.csv
```

Result:

```text
SONIC_G1_DEPLOYMENT_DRY_RUN_MODE log_replay
QPOS_SOURCE /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/planner_ort_walk_qpos.csv
RAW_ACTION_SOURCE /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_walking_raw_actions_log_300f.csv
TOKEN_SOURCE zero
FRAMES 64
OBS_DIM 994
OBS_FINITE True
RAW_ACTION_FINITE True
TARGET_FINITE True
ROOT_Z_MIN 0.727747798
ROOT_Z_MAX 0.787849367
RAW_ACTION_MAX_ABS 6.32903862
TARGET_MAX_ABS 2.01109475
OUTPUT_SUMMARY_CSV /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/sonic_g1_deployment_dry_run_planner_qpos_official_actions_64f.csv
SONIC_G1_DEPLOYMENT_DRY_RUN_OK
```

Summary CSV inspection:

```text
SUMMARY_ROWS 64
SUMMARY_COLS 70
ROW0_ROOT_Z 0.787849367
ROW0_OBS_DIM 994
ROW0_OBS_FINITE True
ROW0_RAW_ACTION_MAX_ABS 0
ROW0_TARGET_MAX_ABS 0.669
ROW63_ROOT_Z 0.727747798
ROW63_TARGET_MAX_ABS 1.17932243
```

# Review

Status: pass.

This validates the dry-run bridge on H200 without Genesis stepping or hardware
commands. It is intentionally a replay I/O check, not a closed-loop policy
rollout and not a real-time test.
