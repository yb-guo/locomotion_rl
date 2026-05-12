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
- 2026-05-12 Router local extended verification:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_g1_curriculum_ppo_smoke.py
  tests\test_g1_policy_action_safety_probe.py tests\test_g1_ppo_smoke.py
  tests\test_ppo_loop.py -q -p no:cacheprovider` -> 22 passed, 9 skipped.
- 2026-05-12 H200 focused verification through guarded command:
  `PYTHONPATH=src python -m pytest tests/test_g1_velocity_tracking_env.py
  tests/test_ppo_loop.py tests/test_g1_ppo_smoke.py -q -p no:cacheprovider`
  -> 24 passed in 2.70s.
- 2026-05-12 H200 extended compatibility verification through guarded command:
  `PYTHONPATH=src python -m pytest tests/test_g1_curriculum_ppo_smoke.py
  tests/test_g1_policy_action_safety_probe.py tests/test_g1_ppo_smoke.py
  tests/test_ppo_loop.py -q -p no:cacheprovider` -> 31 passed in 2.46s.
- 2026-05-12 H200 standing PPO reset-metrics run through guarded command:
  `CUDA_VISIBLE_DEVICES=1`, physical GPU 1, logical `cuda:0`,
  `command_mode=standing`, `action_scale_mult=0.25`, `root_z=1.20`,
  `termination_height_min=0.20`, run id
  `h200-gpu1-standing-reset-metrics-v1`. Result: status ok, 3/3 seeds passed,
  min collect throughput 35292.82 env-policy steps/s, mean final reward
  1.63395, mean final episode_length_mean 51.9209, mean final survival_rate
  1.0, max final height_reset_rate 0.0, max final tilt_reset_rate 0.0, max final
  timeout_rate 0.0, any_final_full_env_reset_wave false. Final per-seed
  `completed_episode_count` remained 0 because no env reached a reset/timeout
  in the final rollout.

## Review

- 2026-05-12 Read-only reviewer found no implementation correctness blocking
  issue. Initial P1 finding was evidence gap only; H200 evidence above resolves
  the subtask003 acceptance gap. Residual risk: cause counts are diagnostic
  flag counts over env-steps and are not mutually exclusive reset buckets.

- 2026-05-12 Re-review found no blocking findings. Reviewer agreed the H200
  evidence resolves the prior P1 evidence gap and that subtask003 can be
  considered complete without marking task020 passed.

Status: complete.
