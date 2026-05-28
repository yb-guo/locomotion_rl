# 005 LocoFormer-Style Consumer

## Route

Implement a minimal tokenized history consumer on top of the shared history
buffer.

Scope:

- This is not full LocoFormer morphology generalization.
- Fixed G1-like topology only.
- Use the existing 31D action and existing eval tasks.
- Tokenization may include joint/body/time tokens derived from actor-visible
  history only.

Non-goals:

- no random morphology;
- no link mass/COM/inertia variation;
- no new simulator;
- no separate eval harness.

## Log

- 2026-05-28 Planned as third policy consumer.

## Review

Status: planned. Pass requires tokenization smoke, overhead JSON, and at least a
minimal policy-forward smoke.
