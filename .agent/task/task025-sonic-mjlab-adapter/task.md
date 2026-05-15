# Task025: SONIC mjlab Adapter

## Goal

Run GEAR-SONIC as an external controller inside the current `unitree_rl_mjlab`
MuJoCo/ManagerBasedRlEnv stack and render rollout evidence.

This is adapter work, not training. The first version must not modify the
RSL-RL runner or the mjlab task implementation.

## Route

1. Add a backend adapter that exposes mjlab G1 as the existing
   `G1RobotBackend` contract.
2. Reuse the existing SONIC scalar runtime boundary:
   `ActionProvider -> ScalarG1Runtime -> G1MotorCommand -> G1RobotBackend`.
3. Keep planner/encoder/decoder runtime isolated behind small wrappers.
4. Start with zero/sequence action smokes before online planner rollout.
5. Record numeric evidence and video/contact-sheet output when H200 execution
   succeeds.

## Subtasks

- `001-contract-audit.md`
- `002-mjlab-backend-smoke.md`
- `003-sonic-sequence-replay.md`
- `004-online-planner-encoder-rollout.md`
- `005-review.md`
- `006-sonic-dependency-setup.md`

## Acceptance

- mjlab backend reads finite 29DoF G1 state in SONIC MuJoCo command order.
- backend maps SONIC motor targets to mjlab joint-position actions by joint
  name, not by assumed index.
- zero-action smoke can step mjlab without non-finite state.
- SONIC sequence or online rollout records finite actions, root z, displacement,
  and rendered video or contact sheet.
- Failures document whether the blocker is joint order, action scale/default
  offset, motor gains, asset mismatch, or SONIC runtime.

## Log

- 2026-05-15 Opened task after task024 confirmed `unitree_rl_mjlab`
  `Unitree-G1-Flat` can render current RSL-RL checkpoint videos.
- 2026-05-15 Remote mjlab inspection found `joint_pos.target_names` matches the
  SONIC 29DoF command MuJoCo order:
  left leg, right leg, waist, left arm, right arm.
- 2026-05-15 H200 synthetic sequence smoke rendered a 120-step video with no
  termination. This verified CSV sequence plumbing only; it was not an official
  SONIC replay and the robot fell under repeated fixture actions.
- 2026-05-15 Added a `sonic` optional Python dependency extra for adapter-side
  ONNX helper code: `numpy` and `onnx`. H200 `unitree-rl-mjlab` already has
  these Python modules installed; Python `onnxruntime` is not required for the
  current adapter path.

## Review

Status: partial implementation.

Implemented the modular backend/controller boundary and verified local unit
tests. H200 zero-action and synthetic-sequence mjlab smokes executed and
rendered, but both collapsed because neither uses a stabilized SONIC action
trace. Official sequence/online rollout is blocked by missing SONIC artifacts
in the current H200 workspace.

Verification:

- `PYTHONPATH=src python -m pytest -p no:cacheprovider`
  passed: `307 passed, 17 skipped`.
- `python -m ruff check ...` was not run because `ruff` is not installed in the
  local Python environment.
