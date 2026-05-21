# 005: Dynamic Eval Grid And Render

## Route

Evaluate trained dynamic MLP checkpoints and expand speed to `2.0 m/s`.

Per-speed gates:

- fixed command clean eval
- persistent failure eval
- dynamic single-failure eval
- dynamic switch eval
- per-joint dynamic onset grid
- switch grid
- render clean, dynamic single, and dynamic switch videos

Speed ladder:

1. Close `1.6 m/s`.
2. Extend to `1.8 m/s`.
3. Extend to `2.0 m/s`.

Dynamic pass thresholds:

- `zero_fall_ratio >= 0.90`
- `recovery_success_ratio >= 0.75`
- post-recovery `lin_vel_error_mean <= 0.8`
- post-recovery `yaw_vel_error_mean <= 0.8`
- `max_gravity_xy_after_onset <= 0.8`

Pass:

- Every speed stage has JSON summaries and video evidence.
- Render review does not show stop-walking, excessive shaking, dragging, or
  upper-body flailing as the adaptation mechanism.
- `2.0 m/s` is accepted only after `1.6` and `1.8` are closed.

Fail:

- The speed ladder skips directly to `2.0 m/s`.
- Dynamic switch cases are omitted.
- Videos are missing for the final accepted checkpoint.

## Log

- 2026-05-21 Opened.

## Review

Status: open.
