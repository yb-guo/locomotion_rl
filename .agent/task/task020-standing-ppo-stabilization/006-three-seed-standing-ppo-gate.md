# 006: Three Seed Standing PPO Gate

## Goal

Run the standing PPO stabilization gate with selected config.

## Route

1. H200 physical GPU 1.
2. 3 seeds.
3. `command_mode=standing`.
4. Suggested gate:
   - `n_envs=1024`;
   - `rollout_steps=32`;
   - `ppo_updates=20`;
   - `epochs=2`;
   - `minibatch_size=8192`.
5. Record train metrics and summary.

## Acceptance

- All success metrics in task contract pass.
- No NaN/Inf.
- Actor/value params change.
- No full-env reset wave every rollout.
- Episode length reaches `>= 200` or `>= 2x` baseline.
- Readiness for deterministic standing eval is clear.

## Log

- 2026-05-12 Planned.

## Review

Status: pending.
