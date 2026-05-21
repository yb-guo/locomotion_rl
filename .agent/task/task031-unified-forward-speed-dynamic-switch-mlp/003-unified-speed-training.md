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

## Review

Status: planned. Pass requires clean and persistent failure eval passing at
`0.4`, `0.8`, `1.2`, `1.6`, and `2.0 m/s`, plus checkpoint and JSON paths.
Per-case thresholds are `zero_fall_ratio >= 0.90`,
`lin_vel_error_mean <= 0.8`, `yaw_vel_error_mean <= 0.8`, and
`gravity_xy_mean <= 0.8`. Persistent eval includes both the Task029 random
motor-only failure distribution and the 12-joint forced persistent dead-motor
grid.
