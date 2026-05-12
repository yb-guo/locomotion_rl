# 006b: Deterministic Standing Eval

## Goal

Verify learned standing policy outside noisy PPO rollout collection.

## Route

1. Load checkpoint from subtask006.
2. Run deterministic action as actor mean.
3. Use standing commands only.
4. Suggested eval:
   - `n_envs=256`;
   - `steps=512`;
   - no PPO update.
5. Record survival curve, reset cause, root height, upright, action saturation,
   and episode length.

## Acceptance

- Eval confirms train metric improvement.
- No claim relies only on stochastic train rollout.
- If eval fails, do not open yaw/vx.

## Log

- 2026-05-12 Planned.

## Review

Status: pending.
