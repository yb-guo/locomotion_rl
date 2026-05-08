# Task 008: Runtime Profile Foundation

## Goal

Create the profile/action-bridge foundation for efficient train/inference unity.

The first target remains Unitree G1 29DoF with the official SONIC control
profile. The architecture must leave room for future LocoFormer-style
multi-embodiment work, but this task does not implement multi-robot training.

## Scope

- Add a YAML robot/control profile for SONIC G1 29DoF.
- Add typed profile schema, loader, validation, and compiled profile objects.
- Move runtime action bridge authority from Python constants to loaded YAML.
- Provide scalar and tensor action bridge implementations.
- Prove scalar/tensor action bridge equivalence.
- Keep task006/task007 routes working.

## Non-Goals

- No full `SonicG1ScalarRuntime` extraction yet.
- No PPO training loop.
- No LocoFormer multi-embodiment training.
- No real G1 command publishing.
- No dexterous hand control.
- No new datasets, checkpoints, assets, or upstream repos.

## Decisions

- Runtime reads YAML from initialization time.
- `pyyaml` becomes a core dependency.
- Hot training loops must use compiled profiles, not YAML or joint-name lookup.
- YAML uses multiple arrays rather than one joint object per row.
- Control arrays are stored in command/MuJoCo order.
- Raw policy actions remain in policy/IsaacLab order.

## Subtasks

- `001-yaml-robot-profile.md`
- `002-profile-loader-validation.md`
- `003-scalar-action-bridge.md`
- `004-tensor-action-bridge.md`
- `005-migrate-sonic-bridge.md`
- `006-h200-smoke.md`

## Acceptance

- Local tests pass.
- H200 targeted tests pass.
- Scalar action bridge matches current official SONIC bridge for representative
  actions.
- Tensor action bridge with batch size 1 matches scalar output.
- Tensor action bridge supports batched raw actions without per-frame dataclass
  allocation.
- Existing task006/task007 H200 smoke paths do not regress.

## Task Discipline

Every subtask must be a minimum closed loop:

- add or change one narrow capability;
- include local tests or a deterministic smoke command;
- record concrete evidence in that subtask's `Log`;
- keep `Review` at pending/in progress until verification exists;
- do not merge pure scaffolding without an exercised path.

## Result

Status: passed.

Closed scope:

- YAML robot/control profile for the Unitree G1 29DoF SONIC body profile.
- Initialization-time profile loader, validation, and compiled profile object.
- Profile-backed scalar action bridge.
- Profile-backed NumPy tensor action bridge for batched action mapping.
- Existing SONIC G1 action bridge migrated to the compiled profile authority.
- Existing task006/task007 routes kept compatible.

Verification evidence:

- Local full suite passed: `PYTHONPATH=src python -m pytest` reported
  `109 passed`.
- H200 targeted full suite passed from `/tmp` against extracted task008 code
  with explicit `PYTHONPATH`: `109 passed in 0.87s`, `PYTEST_STATUS 0`.
- H200 task007 representative dry-run passed:
  `SONIC_G1_DEPLOYMENT_DRY_RUN_OK`, `OBS_DIM 994`, `OBS_FINITE True`,
  `RAW_ACTION_FINITE True`, `TARGET_FINITE True`.
- Read-only review found no blocking issues after the dry-run command was
  recorded.

Remaining non-goals:

- Full `SonicG1ScalarRuntime` extraction is not implemented in task008.
- Vectorized Genesis training backend is not implemented in task008.
- PPO reward, termination, reset curriculum, and large-scale RL throughput are
  not implemented in task008.
- LocoFormer multi-embodiment training is not implemented in task008.
- Real Unitree G1 low-state/low-command publishing is not implemented in
  task008.

## Review

Status: passed.

- 2026-05-08: Top-level task closure reconciled the passed subtask evidence
  with the explicit non-goals. Task008 is complete as a runtime profile/action
  bridge foundation, not as a full backend/runtime or RL training task.
