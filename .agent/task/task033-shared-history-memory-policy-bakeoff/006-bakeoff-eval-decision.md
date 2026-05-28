# 006 Bakeoff Eval Decision

## Route

Compare the three memory consumers using shared eval criteria.

Required first eval:

- blocker subset:
  - speeds `0.4`, `1.6`, `2.0 m/s`;
  - forced persistent dead-grid;
  - canonical dynamic switch.

Escalate to full eval only for consumers that improve the subset:

- speeds `0.4`, `0.8`, `1.2`, `1.6`, `2.0 m/s`;
- clean;
- random persistent motor failure;
- forced persistent 12-joint dead-grid;
- canonical dynamic switch;
- Level C arbitrary onset diagnostic only if blocker subset improves.

Decision outputs:

- overhead table;
- eval table;
- best checkpoint per consumer;
- recommendation for next task.

## Log

- 2026-05-28 Planned as final Task033 review.

## Review

Status: planned. Do not pick a winner without comparable overhead and eval JSON.
