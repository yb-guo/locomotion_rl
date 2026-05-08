# Task 007: SONIC G1 29DoF Deployment Bridge

## Goal

Prepare the current 29-motor SONIC Genesis rollout path for transfer-oriented
dry-run and later real G1 integration.

The immediate goal is not to command hardware. It is to isolate the robot I/O
boundary so the same SONIC planner/encoder/decoder loop can consume either:

- Genesis simulated state;
- recorded state logs;
- eventually Unitree G1 low-level state.

## Scope

- Define a simulator-independent 29DoF G1 robot backend contract.
- Adapt the current Genesis G1 backend to that contract without changing the
  validated Genesis rollout behavior.
- Add a log replay backend for hardware-free dry-run validation.
- Add a dry-run tool that writes per-frame observation/action/target summaries.
- Keep dexterous hand control out of this task; this route is body 29DoF only.

## Non-Goals

- No real robot command publishing.
- No SDK installation.
- No dexterous hand action path.
- No PPO or LocoFormer training changes.
- No new checkpoints, assets, datasets, or upstream repos.

## Subtasks

- `001-backend-dry-run.md`
- `002-h200-dry-run-evidence.md`
- `003-real-g1-lowstate-lowcmd-plan.md`

## Current Inputs

- Primary SONIC/Genesis route: task006 official-context 200-frame pass.
- Body policy action dimension: 29.
- Current transfer-oriented convention:
  `initial_context_source=initial_joint_csv + replan_context_source=motion`.

## Review

Status: L1 backend dry-run scaffold implemented and H200 dry-run passed.

Implemented:

- backend-neutral `G1RobotState`, `G1MotorCommand`, and `G1RobotBackend`
  contract;
- `GenesisG1SceneRobotBackend` wrapper for the existing Genesis scene backend;
- `LogReplayG1RobotBackend` for hardware-free 36D qpos replay;
- `sonic_g1_deployment_dry_run` summary tool;
- planner qpos extraction in the task006 online rollout now goes through the
  backend-neutral robot state helper.

Verification:

```text
PYTHONPATH=src python -m pytest -p no:cacheprovider
76 passed
```

Local `ruff` was not run because this machine does not have `ruff` installed:

```text
No module named ruff
```

H200 dry-run:

```text
QPOS_SOURCE /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/planner_ort_walk_qpos.csv
RAW_ACTION_SOURCE /root/h200-locomotion-lab-runs/task006-sonic-genesis-action-policy/actions/official_walking_raw_actions_log_300f.csv
FRAMES 64
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

Next route: formalize the real G1 `LowState -> RobotBackend -> LowCmd`
adapter plan before any hardware command path is implemented.
