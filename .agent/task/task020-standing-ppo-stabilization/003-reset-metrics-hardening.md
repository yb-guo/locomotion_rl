# 003: Reset Metrics Hardening

## Goal

Make reset semantics trainable and measurable before reward tuning.

## Route

1. Keep `height_min` as diagnostic/reward metric.
2. Keep hard termination height at fall semantics, default `0.20`.
3. Track reset causes separately:
   - `tilt_reset_count`;
   - `height_reset_count`;
   - `timeout_count`;
   - `full_env_reset_wave`.
4. Add episode length and survival metrics.
5. Verify height reset does not dominate standing PPO.

## Acceptance

- `height_bad` and hard `termination_height_bad` are not conflated.
- Metrics report reset cause and episode length per update.
- H200 run shows reset causes are interpretable.
- No reward/action tuning happens before this is verified.

## Log

- 2026-05-12 Planned.

## Review

Status: pending.
