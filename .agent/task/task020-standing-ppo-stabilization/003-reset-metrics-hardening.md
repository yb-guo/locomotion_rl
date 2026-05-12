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
- 2026-05-12 Implemented local reset metric hardening:
  `G1VelocityTrackingVectorizedEnv.step` now exports pre-reset
  `episode_lengths`, `completed_episode_lengths`, and `full_env_reset_wave`.
  `collect_rollout` now carries episode length, completed episode length,
  height/tilt reset counts, and full-env wave flags into `RolloutBatch`.
  `g1_ppo_smoke` now emits reset rates, timeout rate, survival rate, episode
  length stats, completed episode stats, and full-env reset wave metrics per
  update, plus final-seed aggregate reset/survival metrics in `summary.json`.
  No reward/action tuning was changed.
- 2026-05-12 Local focused verification:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_g1_velocity_tracking_env.py
  tests\test_ppo_loop.py tests\test_g1_ppo_smoke.py -q -p no:cacheprovider`
  -> 19 passed, 5 skipped in 0.20s.

## Review

Status: local implementation verified; H200 evidence and read-only review pending.
