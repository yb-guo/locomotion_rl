# 004: H200 PPO-Update Isolation

## Goal

Only if no-update probes are stable, run a minimal PPO-update isolation matrix
to distinguish actor update from value/reward/advantage effects.

## Route

1. Confirm subtask 003 permits this route.
2. Run minimal standing-only u50 variants such as:
   - normal PPO baseline;
   - actor-frozen or critic-only update;
   - deterministic collection with actor update disabled, if available.
3. Compare reset waves, actor/value parameter changes, KL, action stats, and
   reset counts.

## Log

- 2026-05-11 Skipped by stop rule.

Reason:

- Subtask 003 showed `zero_action` no-update already falls:
  `first_tilt_chunk=2`, `max_reset_count=1024`,
  `mean_reset_count=348.16`, `final_reset_count=1024`, and
  `final_tilt_bad_count=1024`.
- Because the zero-action no-update path is unstable, task018 does not have
  evidence that PPO updates are the first causal boundary. Actor-frozen or
  critic-only PPO isolation would be misleading here.

No PPO-update isolation commands were run.

## Review

Status: skipped by stop rule.

- Final read-only reviewer agreed the skip is correct because `zero_action`
  no-update already falls.
