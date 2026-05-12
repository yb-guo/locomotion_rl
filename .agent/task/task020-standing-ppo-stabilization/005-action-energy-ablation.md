# 005: Action Energy Ablation

## Goal

Find early standing action energy that can learn without triggering full reset
waves.

## Route

1. Keep reward/reset config fixed from previous subtasks.
2. Run small H200 matrix:
   - `action_scale_mult`: `0.10`, `0.20`, `0.25`, `0.35`;
   - `log_std_init`: `-2.0`, `-1.5`, `-1.0`.
3. Record:
   - reset rate;
   - episode length;
   - action saturation ratio;
   - `log_std_mean/min/max`;
   - KL and clip fraction;
   - actor/value param delta.
4. Choose smallest action energy that still changes policy params and improves
   survival.

## Acceptance

- Matrix is bounded.
- One candidate is selected with evidence, or env/contact blocker is declared.
- No yaw/vx is introduced.

## Log

- 2026-05-12 Planned.

## Review

Status: pending.
