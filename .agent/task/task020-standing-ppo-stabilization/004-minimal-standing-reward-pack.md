# 004: Minimal Standing Reward Pack

## Goal

Create a minimal standing reward that promotes active balance without walking
or reward hacking.

## Route

1. Keep command tracking disabled or neutral for standing.
2. Add/verify components:
   - upright reward;
   - base height target reward;
   - joint velocity penalty;
   - action rate penalty;
   - default pose deviation penalty;
   - small alive reward;
   - termination penalty.
3. Record component scales every update.
4. Reject reward if reward improves while survival worsens.

## Acceptance

- Reward components are finite and comparable.
- Root height does not learn permanent low crouch.
- Action saturation does not rise with reward.
- Survival or episode length improves in H200 smoke.

## Log

- 2026-05-12 Planned.

## Review

Status: pending.
