# 003 Unified Speed Training

## Route

Train the Level A unified-speed MLP from `model_5349.pt`.

Implementation target:

- Add or patch a H200 MJLab env config that samples forward speed continuously
  from `0.4..2.0 m/s`.
- Keep clean and persistent motor-failure rehearsal in the training
  distribution.
- Preserve Task029/Task030 failure semantics and action/observation contracts.
- Do not introduce explicit actor fault labels or speed-bin labels.

Training should be staged only if needed by baseline evidence. The first
attempt should prefer a direct unified-speed rehearsal rather than creating many
fixed-speed one-off configs.

## Log

- 2026-05-21 Planned as Level A training after baseline eval.
- 2026-05-21 Added local launch artifact
  `artifacts/task031_launch_unified_persistent_from5349.sh`. It uses the
  registered Task031 unified persistent env and symlinks Task030 `model_5349.pt`
  into the Task031 experiment before launching RSL-RL resume.
- 2026-05-21 Level A generic unified persistent run completed from `model_5349`
  to `model_5428`, but eval showed remaining dead-grid failures and a `2.0 m/s`
  dynamic-switch regression. Early checkpoint `model_5350` preserved dynamic
  switch but still failed low-speed dead-grid and `2.0 m/s` right-knee dead.
- 2026-05-21 Added focused dead-grid guard patch artifact
  `artifacts/task031_create_focused_deadgrid_stage.py` to oversample observed
  failing single-dead joints across `0.4..2.0 m/s`.
- 2026-05-21 Focused guard evals for `model_5355` and `model_5360` preserved
  dynamic-switch behavior on the subset tested, but did not close forced
  persistent dead-grid. Remaining failures included low-speed
  `left_hip_yaw_joint`/`left_hip_roll_joint` and high-speed `right_knee_joint`.

## Review

Status: failed under the original Level A forced dead-grid gate. Pass would
require clean and persistent failure eval passing at
`0.4`, `0.8`, `1.2`, `1.6`, and `2.0 m/s`, plus checkpoint and JSON paths.
Per-case thresholds are `zero_fall_ratio >= 0.90`,
`lin_vel_error_mean <= 0.8`, `yaw_vel_error_mean <= 0.8`, and
`gravity_xy_mean <= 0.8`. Persistent eval includes both the Task029 random
motor-only failure distribution and the 12-joint forced persistent dead-motor
grid. The current evidence shows the forced grid remains the blocker.
