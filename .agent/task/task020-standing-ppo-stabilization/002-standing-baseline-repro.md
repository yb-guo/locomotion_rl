# 002: Standing Baseline Repro

## Goal

Run the current PPO stack in standing mode without tuning, to establish the
baseline failure or baseline stability.

## Route

1. Use current `g1_ppo_smoke`/PPO stack.
2. Run `command_mode=standing`.
3. Use conservative starting knobs:
   - `action_scale_mult=0.25`;
   - `root_z=1.20`;
   - `termination_height_min=0.20`;
   - `default_pose=tall_crouch`.
4. Record reset cause, episode length, reward, KL, entropy, grad norm, value
   loss, throughput, and device report.
5. Stop if NaN/device/throughput fails.

## Acceptance

- H200 baseline run exists.
- Baseline summary records standing-only config.
- Baseline identifies whether failure is PPO plumbing, reset semantics, reward,
  action energy, or env/contact.

## Log

- 2026-05-12 Planned.

## Review

Status: pending.
