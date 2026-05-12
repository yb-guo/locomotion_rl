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

## Review

Status: pending.
