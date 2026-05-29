# 005 TXL-Style Memory Consumer

## Route

Add a long-context policy consumer only after the multi-trial contract and eval
semantics are stable.

Acceptance criteria:

- Target context horizon is `3.2s = 160` policy steps at `50Hz`.
- TXL-style memory preserves state across inner trial reset.
- TXL-style memory clears state on outer episode reset.
- Actor does not see trial labels or failure debug labels.
- env64 inference and one PPO iteration run.
- env8192 overhead is recorded.
- No locomotion-quality claim is made from construction smoke.

## Log

- 2026-05-29 Planned.

## Review

Status: pending.
