# Route

Task: task004-genesis-g1-baseline

Goal: Add the first PPO baseline loop.

Scope:

- `src/h200_locomotion_lab/training`
- configs under `configs/experiments`
- focused tests for config and rollout shape

Verify:

- One tiny rollout runs.
- Loss computes.
- Checkpoint/log path is explicit.

Environment:

- H200 target for real training
- local shape tests allowed

No Hack:

- no fake reward improvement
- no hard-coded device
- no silent NaN handling

Hardware:

- avoid per-env Python loops in hot path
- record env count and steps per second

# Log

# Review

Result: pending
Syntax:
Hack:
Scope:
Efficiency:
Hardware:
Verify:
Findings:

