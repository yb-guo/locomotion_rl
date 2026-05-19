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
- 2026-05-12 Local related PPO/env verification:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_g1_action_energy_ablation.py tests\test_g1_velocity_tracking_env.py tests\test_g1_curriculum_ppo_smoke.py tests\test_g1_policy_action_safety_probe.py tests\test_g1_ppo_smoke.py tests\test_ppo_loop.py -q -p no:cacheprovider`
  -> 48 passed, 10 skipped.
- 2026-05-12 Read-only re-review found no blocking issue after the count fix.
- 2026-05-12 H200 focused verification after syncing commit `af4a0c4`:
  `PYTHONPATH=src python -m pytest tests/test_g1_action_energy_ablation.py tests/test_g1_velocity_tracking_env.py tests/test_g1_curriculum_ppo_smoke.py tests/test_g1_policy_action_safety_probe.py tests/test_g1_ppo_smoke.py tests/test_ppo_loop.py -q -p no:cacheprovider`
  -> 58 passed.
- 2026-05-12 Re-ran the same 3-seed standing gate as
  `h200-gpu1-standing-gate-v2` with the new summary fields. Result:
  `status=ok`, all seeds passed plumbing checks, physical GPU 1, logical
  `cuda:0`, min collect throughput 44088.53 env-policy steps/s, final
  mean episode_length_mean 67.291992, max training episode_length_mean
  71.383545, max training reset_rate 0.03125, and max training
  tilt_reset_rate 0.03125.
- 2026-05-12 The new full-env reset-wave fields clarify the failure mode:
  `any_training_full_env_reset_wave=false` and
  `training_full_env_reset_wave_count=0`, so there was no single simulation
  step where every env reset together. However, every seed still had
  `reset_count=1024` and `tilt_reset_count=1024` in updates 2, 5, 8, 11, 14,
  and 17. Since `n_envs=1024` and `rollout_steps=32`, this means all envs fell
  once within those rollout windows, just not on the same step.

## Review

Status: complete. Read-only re-review found no blocking issue after the count
fix. H200 v2 evidence shows the summary now separates single-step full-env
reset waves from rollout-window reset sweeps.
