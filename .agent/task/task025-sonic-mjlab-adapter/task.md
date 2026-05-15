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
- `007-alignment-diagnosis.md`
- `008-command-and-actuator-alignment.md`
- `009-ankle-pitch-residual.md`

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
- 2026-05-15 Ran longer 400-step and 800-step online rollouts with normal mjlab
  terminations enabled. Neither triggered `fell_over`; the 800-step run moved
  about 7.96 m and ended with root z about 0.738. Contact sheet inspection
  shows the robot remains upright enough to step, but with a low, crouched,
  backward-leaning posture.
- 2026-05-15 Added alignment diagnosis after the 800-step visual review. The
  strongest immediate mismatch is mjlab HOME reset/action offset versus SONIC
  crouched default angles; hip-pitch action scale/controller constants also
  differ between mjlab and the SONIC profile.
- 2026-05-15 Added a deterministic alignment trace CLI and H200 ablations. The
  reset-only, startup-randomization-only, naive explicit-height, and unseeded
  velocity-tweak hypotheses did not hold up.
- 2026-05-15 Added `planner_context_source={live,motion}` to the online SONIC
  provider. On H200 with `seed=123`, 400-step fixed-base A/B improved from
  live-context `root_z_final=0.7049`, `abs_pitch_p95=0.5607` to motion-context
  `root_z_final=0.7598`, `abs_pitch_p95=0.2627`, with no terminations.
- 2026-05-15 Added planner/root velocity instrumentation and planner-only
  command sweep. `target_vel` is not a signed metric velocity for SONIC mode 2:
  direction comes from `movement_direction`, while `target_vel=0.5` gives a
  slower gait bucket than `-1.0/-0.5/1.0`.
- 2026-05-15 On H200 with motion context, `target_vel=0.5` reduced the 400-step
  trace from `abs_pitch_p95=0.2627`, `joint_error_rms_mean=0.2606` to
  `abs_pitch_p95=0.1827`, `joint_error_rms_mean=0.1703`.
- 2026-05-15 Confirmed mjlab G1 hip-pitch actuator profile is 7520_14
  (`kp=40.18`, `effort=88`) while SONIC profile uses 7520_22 (`kp=99.10`,
  `effort=139`). A trace-only hip-pitch profile override improved the 400-step
  `target_vel=0.5` run to `abs_pitch_p95=0.1229`,
  `joint_error_rms_mean=0.1528`.
- 2026-05-15 Added ankle-focused trace instrumentation. In the current best
  H200 400-step baseline, ankle pitch residual is not explained by actuator
  force saturation or actual joint-limit clipping. The actual ankle joints stay
  inside soft limits, while the SONIC motor targets sometimes command outside
  mjlab soft limits. During target violations, left/right ankle-pitch RMS error
  rises to about `1.15` / `0.95` rad versus `0.28` / `0.32` rad when targets
  remain inside limits.

## Review

Status: adapter implementation passed; stable locomotion remains open.

Implemented the modular backend/controller boundary and verified local unit
tests. H200 zero-action and synthetic-sequence mjlab smokes executed and
rendered, but both collapsed because neither uses a stabilized SONIC action
trace. The C++ planner runner dependency is now buildable on H200. Official
SONIC ONNX artifacts were restored and the true online planner/encoder/decoder
path now runs and renders in `unitree_rl_mjlab`.

The remaining risk is policy/context quality, not adapter availability. The
deterministic traces now identify three concrete alignment issues:

- replanning from live qpos history instead of previous planner motion;
- using `target_vel=-1.0` as if it were a signed metric velocity;
- mjlab's hip-pitch actuator profile being weaker than the SONIC profile.

The current best diagnostic baseline is motion context, `target_vel=0.5`, and
the SONIC hip-pitch actuator profile. That still leaves ankle pitch as the
dominant residual tracking error. The first ankle probe points to target/limit
contract mismatch rather than force saturation: SONIC sometimes asks for ankle
targets outside mjlab soft limits, while the actual joints remain inside those
limits. The next route should compare official SONIC ankle limits with mjlab G1
limits, then run a trace-only target clamp probe before filling optional encoder
fields.

Verification:

- `PYTHONPATH=src python -m pytest -p no:cacheprovider`
  passed: `319 passed, 17 skipped`.
- `PYTHONPATH=src python -m pytest tests/test_sonic_controller.py
  tests/test_mjlab_sonic_alignment_trace.py -q`
  passed: `7 passed` with only a local pytest cache permission warning.
- H200 `seed=123`, 400-step fixed-base trace:
  - live context: no done, `root_z_final=0.7049`, `abs_pitch_p95=0.5607`.
  - motion context: no done, `root_z_final=0.7598`, `abs_pitch_p95=0.2627`.
- H200 `seed=123`, 400-step motion-context command/profile traces:
  - `target_vel=0.5`: no done, `abs_pitch_p95=0.1827`,
    `joint_error_rms_mean=0.1703`.
  - `target_vel=0.5` plus SONIC hip-pitch actuator profile: no done,
    `abs_pitch_p95=0.1229`, `joint_error_rms_mean=0.1528`.
- H200 `seed=123`, 400-step ankle-focused trace on the current best baseline:
  no done, `abs_pitch_p95=0.1225`, `joint_error_rms_mean=0.1527`. Actual
  soft-limit violation fraction was `0.0` for all joints; target soft-limit
  violation fraction was `0.0925` for left ankle pitch and `0.0700` for right
  ankle pitch.
- `python -m ruff check ...` was not run because `ruff` is not installed in the
  local Python environment.
