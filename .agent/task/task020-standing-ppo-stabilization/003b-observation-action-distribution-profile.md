# 003b: Observation Action Distribution Profile

## Goal

Check whether PPO instability comes from scale/pathology in observations,
actions, rewards, values, or advantages before changing reward.

## Route

1. Record observation mean/std/min/max.
2. Record action mean/std/min/max and saturation ratio.
3. Record reward component scale.
4. Record value prediction, return, and advantage mean/std/min/max.
5. Record `log_std_mean`, `log_std_min`, `log_std_max`.
6. Decide whether normalization or std constraints are needed.

## Acceptance

- H200 profile artifact exists.
- No normalization is added without profile evidence.
- If profile is sane, continue to reward pack.
- If profile is pathological, stop and fix scale first.

## Log

- 2026-05-12 Planned.
- 2026-05-12 Implemented local metrics plumbing for standing PPO profile:
  observation/action/reward/value/return/advantage distribution stats,
  action saturation ratio, `log_std` summary, and available reward component
  means. Local focused tests passed:
  `PYTHONPATH=src python -m pytest tests/test_ppo_loop.py tests/test_g1_ppo_smoke.py -q -p no:cacheprovider`
  (`12 passed, 6 skipped`).
- 2026-05-12 H200 profile artifact not generated in this subtask run; task
  remains pending H200 execution evidence.
- 2026-05-12 Read-only review found no blocking implementation issue and raised
  one P2 diagnostic issue: raw reward component means could mislead reward scale
  diagnosis because env reward applies weights and penalty signs. Fixed by
  keeping raw `reward_component_*_mean` and adding weighted/signed
  `reward_contribution_*_mean`.
- 2026-05-12 Local focused verification after contribution fix:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_ppo_loop.py
  tests\test_g1_ppo_smoke.py -q -p no:cacheprovider` -> 13 passed, 6 skipped.
- 2026-05-12 H200 focused verification through guarded command:
  `PYTHONPATH=src python -m pytest tests/test_ppo_loop.py
  tests/test_g1_ppo_smoke.py -q -p no:cacheprovider` -> 19 passed in 2.75s.
- 2026-05-12 H200 extended compatibility verification through guarded command:
  `PYTHONPATH=src python -m pytest tests/test_g1_velocity_tracking_env.py
  tests/test_g1_curriculum_ppo_smoke.py tests/test_g1_policy_action_safety_probe.py
  tests/test_g1_ppo_smoke.py tests/test_ppo_loop.py -q -p no:cacheprovider`
  -> 42 passed in 3.12s.
- 2026-05-12 H200 profile artifact through guarded command:
  `CUDA_VISIBLE_DEVICES=1`, physical GPU 1, logical `cuda:0`,
  `command_mode=standing`, `action_scale_mult=0.25`, `root_z=1.20`,
  `termination_height_min=0.20`, run id `h200-gpu1-standing-profile-v2`.
  Result: status ok, 3/3 seeds passed, min collect throughput 35548.72
  env-policy steps/s, mean final survival_rate 1.0, no final reset wave.
  Final profile ranges across seeds: observation_std about 0.314,
  observation min/max about -3.31/3.36, action_std about 0.476,
  action_saturation_ratio about 0.0026, log_std_mean about -0.499,
  reward_std about 0.298-0.304, return_mean about 15.28-15.34,
  advantage_mean about 14.98-15.05, value_prediction_mean about 0.29-0.32.
  Weighted reward contributions: alive 0.05, lin_vel about 0.668-0.674,
  yaw about 0.426-0.427, upright about 0.492, base_height 0.0,
  action_rate penalty about -0.0045, joint_deviation penalty about -0.00015,
  termination penalty 0.0.
- 2026-05-12 Decision: observation/action/log_std distributions are not
  pathological, and action saturation is not the standing blocker. No
  normalization or std clamp is justified by this profile. Value predictions
  are far below returns in this short smoke, so later longer standing gate
  should watch value loss/return scale, but this does not block reward-pack
  work.
- 2026-05-12 Pulled small H200 evidence files to local ignored task artifacts:
  `.agent/task/task020-standing-ppo-stabilization/artifacts/h200-gpu1-standing-profile-v2/`
  contains `config.json`, `summary.json`, and `metrics.jsonl`. Checkpoints were
  not pulled.

## Review

- 2026-05-12 Read-only reviewer found no blocking implementation issue. P2 raw
  reward component naming/scale issue was fixed with explicit contribution
  metrics.
- 2026-05-12 Re-review found no blocking findings. Reviewer agreed the P2 is
  resolved, H200 profile evidence satisfies subtask003b acceptance, and task020
  remains correctly not passed.

Status: complete.
