# 001: Dynamic Failure Contract

## Route

Define the exact contract before adding code.

Scope:

- Keep current MLP policy and current actor observation/action contract.
- Actor input remains `104` dims and action remains `31` dims.
- Actor must not observe active fault state, motor scale, failure mask, or
  segment id.
- Critic/eval/logging may record fault state.
- First dynamic fault types are `weak motor` and `dead motor`.
- First dynamic faults target leg motors only.
- First pass excludes locked joints, stuck commands, multi-motor dynamic
  failures, online delay jumps, and contact/link/sensor randomization changes.

Dynamic timing contract:

- Eval starts with deterministic templates.
- Training uses randomized onset, duration, joint, and severity.
- Fault onset/switch allows `0.3 s` transient recovery for tracking metrics.
- Falls, base instability, and severe height loss are still counted during the
  transient window.

## Log

- 2026-05-21 Opened with the user-approved Task030 decisions from planning.

## Review

Status: open. This subtask passes only after an inspect artifact proves the
actor/action contract and the dynamic fault settings.
