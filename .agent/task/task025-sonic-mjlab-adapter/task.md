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
- 2026-05-15 Downloaded a Linux CPython 3.11 wheelhouse locally and uploaded it
  to H200 for offline SONIC Python dependency setup:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/sonic_wheelhouse_linux_cp311`.
- 2026-05-15 Downloaded and uploaded ONNX Runtime C++ Linux x64 1.19.2, then
  built `bin/sonic_planner_ort_runner` on H200. The binary links against the
  uploaded runtime and starts correctly.
- 2026-05-15 Checked local agent/download paths before downloading. Official
  SONIC ONNX artifacts were not already present locally.
- 2026-05-15 User explicitly requested local download then upload. Downloaded
  official `nvidia/GEAR-SONIC` artifacts into
  `.external_downloads/gear_sonic_artifacts`, uploaded them to
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/gear_sonic_artifacts`,
  and verified remote SHA256.
- 2026-05-15 H200 true online rollout with official planner/encoder/decoder
  ran for 40 and 160 steps with no terminations and rendered videos. The
  160-step run moved forward about 1.27 m while root height dropped about 9 cm.

## Review

Status: adapter implementation passed; stable locomotion remains open.

Implemented the modular backend/controller boundary and verified local unit
tests. H200 zero-action and synthetic-sequence mjlab smokes executed and
rendered, but both collapsed because neither uses a stabilized SONIC action
trace. The C++ planner runner dependency is now buildable on H200. Official
SONIC ONNX artifacts were restored and the true online planner/encoder/decoder
path now runs and renders in `unitree_rl_mjlab`.

The remaining risk is policy/context quality, not adapter availability: the
160-step online smoke progresses forward but loses height. Next diagnosis
should inspect video frames and compare reset state, context qpos construction,
target command, and mjlab motor gains against official SONIC assumptions.

Verification:

- `PYTHONPATH=src python -m pytest -p no:cacheprovider`
  passed: `307 passed, 17 skipped`.
- `python -m ruff check ...` was not run because `ruff` is not installed in the
  local Python environment.
