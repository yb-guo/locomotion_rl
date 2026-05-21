# 004 Specified Dynamic Switch Training

## Route

Train Level B: unified speed plus the specified Task030 dynamic-switch route.

Implementation target:

- Reuse the Task030 canonical switch as the eval case.
- During training, add small timing jitter to onset and recovery windows so the
  policy does not memorize one exact timestamp.
- Keep the same weak/dead motor semantics used in Task030.
- Keep arbitrary per-joint onset outside the pass target for this subtask.

Evaluation:

- Run canonical dynamic-switch eval at every speed bin:
  `0.4`, `0.8`, `1.2`, `1.6`, and `2.0 m/s`.
- A single-seed candidate must pass all speed bins before final multiseed eval.
- Final acceptance requires `5/5` seeds passing at each of the five speed bins.
- Per-case thresholds are `zero_fall_ratio >= 0.90`,
  `recovery_success_ratio >= 0.75`,
  `post_recovery_lin_vel_error_mean <= 0.8`,
  `post_recovery_yaw_vel_error_mean <= 0.8`, and
  `max_gravity_xy_after_onset <= 0.8`.

## Log

- 2026-05-21 Planned as Level B training after Level A is stable.

## Review

Status: planned. Pass requires canonical dynamic-switch eval passing at all
five speed bins, plus final multiseed evidence.
