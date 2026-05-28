# 003 Stack MLP Consumer

## Route

Implement the smallest memory consumer first: flatten shared history frames and
feed an MLP.

Scope:

- Reuse MJLab/RSL-RL training path if possible.
- Keep action `31D`.
- Keep actor no-fault-label contract.
- Test at least `K=4`; try `K=8` only after overhead is acceptable.

Eval:

- Blocker subset first:
  - speeds `0.4`, `1.6`, `2.0 m/s`;
  - forced persistent dead-grid;
  - canonical dynamic switch.

## Log

- 2026-05-28 Planned as first policy consumer.

## Review

Status: planned. Pass requires smoke/train artifact and blocker-subset eval.
