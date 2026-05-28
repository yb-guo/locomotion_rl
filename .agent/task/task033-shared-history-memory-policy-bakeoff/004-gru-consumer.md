# 004 GRU Consumer

## Route

Implement a recurrent consumer using the same history stream and reset/done
signals.

Scope:

- Do not introduce a second history buffer.
- Hidden state belongs to the policy backend, but reset ids come from the shared
  env/done stream.
- Prefer any existing recurrent support if MJLab/RSL-RL exposes it; otherwise
  document the minimal runner/storage changes required before implementation.

Eval:

- Same blocker subset as StackMLP.
- Same JSON schema plus recurrent metadata: hidden size, sequence length,
  reset handling.

## Log

- 2026-05-28 Planned as second policy consumer.

## Review

Status: planned. Pass requires smoke evidence or a documented runner blocker if
current RSL-RL cannot support recurrent PPO without larger changes.
