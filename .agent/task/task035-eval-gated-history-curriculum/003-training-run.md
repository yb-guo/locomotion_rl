# 003 Training Run

## Route

Run one bounded H200 curriculum training pass.

Constraints:

- env8192 target unless H200 occupancy says otherwise;
- frozen-base StackMLP K4;
- frequent checkpoints for sweep;
- no pass/fail claim from reward curves alone.

Curriculum order:

1. clean + unified-speed rehearsal;
2. weak persistent motor failures;
3. mixed weak/dead single-joint failures;
4. forced dead-grid rehearsal;
5. dynamic-switch rehearsal.

## Log

- 2026-05-28 Planned.

## Review

Status: pending. No training launched yet.
