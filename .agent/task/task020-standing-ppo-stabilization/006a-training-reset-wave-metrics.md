# 006a: Training Reset Wave Metrics

## Goal

Harden the standing PPO gate summary so it cannot miss full-env reset waves
that happen before the final rollout.

## Route

1. Add per-seed training-wide reset-wave aggregates to `g1_ppo_smoke`.
2. Add run-level aggregates across seeds.
3. Keep existing short-smoke plumbing pass semantics separate from task020 gate
   interpretation.
4. Re-run local and H200 focused tests.
5. Re-run or re-read the standing gate with the new summary fields.

## Acceptance

- Summary records whether any training update had a full-env reset wave.
- Summary records the maximum training reset/tilt/height reset rates.
- Existing device/NaN/throughput plumbing checks remain intact.
- Read-only review finds no blocking evidence issue.

## Log

- 2026-05-12 Planned after subtask006 showed final rollout metrics alone were
  insufficient: the H200 `h200-gpu1-standing-gate-v1` metrics had full-env
  tilt reset waves at updates 2, 5, 8, 11, 14, and 17 for every seed, while the
  final rollout had no reset.
- 2026-05-12 Added training-wide per-seed and run-level reset-wave summary
  fields in `g1_ppo_smoke`, plus synthetic helper tests that avoid Genesis.
- 2026-05-12 Local focused test passed:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_g1_ppo_smoke.py -q -p no:cacheprovider`
  -> 14 passed, 1 skipped in 0.72s.
- 2026-05-12 Read-only review found one blocking evidence issue: the first
  implementation counted updates with a full-env reset wave instead of summing
  per-rollout `full_env_reset_wave_count`. Fixed by keeping
  `training_full_env_reset_wave_updates` as update indices but computing
  `training_full_env_reset_wave_count` as the sum of each row's
  `full_env_reset_wave_count`. Added a regression row with three waves in one
  update.
- 2026-05-12 Local focused test after the count fix:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_g1_ppo_smoke.py -q -p no:cacheprovider`
  -> 14 passed, 1 skipped.

## Review

Status: pending re-review after count fix.
