# 006 Long-Context Training Decision

## Route

Train only after the multi-trial reset/memory/eval semantics are proven.

Acceptance criteria:

- Run short training smoke before any longtrain.
- Full quality eval uses speeds `0.4`, `1.2`, and `2.0`.
- Full quality eval includes dynamic switch and forced deadgrid.
- Multi-trial final-trial metrics are the default pass gate.
- Velocity tracking gate prevents stand-still/high-pose local optima.
- Compare against Task036 AdaptK4 partial.
- Decision states promoted, partial, or rejected with JSON evidence.

## Log

- 2026-05-29 Planned.

## Review

Status: pending.
