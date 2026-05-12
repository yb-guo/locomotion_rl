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

## Review

Status: pending.
