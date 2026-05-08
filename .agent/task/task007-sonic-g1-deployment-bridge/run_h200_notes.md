# Task 007 H200 Run Notes

Local pre-H200 verification on 2026-05-08:

```text
PYTHONPATH=src python -m pytest -p no:cacheprovider
76 passed
```

`ruff` is not installed locally:

```text
No module named ruff
```

## H200 Dry-Run 2026-05-08

H200 repo copy:

```text
/root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke
```

Targeted tests:

```text
cd /tmp
PYTHONPATH=/root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke/src \
python3 -m pytest -p no:cacheprovider \
  /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke/tests/test_robot_backend.py \
  /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke/tests/test_sonic_g1_deployment_dry_run.py \
  /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/repo_replay_smoke/tests/test_genesis_sonic_planner_encoder_rollout_probe.py

10 passed
```

Dry-run inputs:

```text
QPOS_SOURCE /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/planner_ort_walk_qpos.csv
RAW_ACTION_SOURCE /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_walking_raw_actions_log_300f.csv
TOKEN_SOURCE zero
FRAMES 64
```

Result:

```text
OBS_DIM 994
OBS_FINITE True
RAW_ACTION_FINITE True
TARGET_FINITE True
ROOT_Z_MIN 0.727747798
ROOT_Z_MAX 0.787849367
RAW_ACTION_MAX_ABS 6.32903862
TARGET_MAX_ABS 2.01109475
SONIC_G1_DEPLOYMENT_DRY_RUN_OK
```

Artifacts:

```text
Log:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/sonic_g1_deployment_dry_run_planner_qpos_official_actions_64f.log

Summary CSV:
/root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/logs/sonic_g1_deployment_dry_run_planner_qpos_official_actions_64f.csv
```
