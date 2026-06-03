# 004: Memory Ablation After Normal Pass

## Route

Only after normal continuous eval passes, evaluate the same checkpoint under
`zero_memory_latent` and `stateless_txl_memory` on the same command, fault
schedule, speed, and seed family.

## Acceptance

- Normal continuous eval passes first.
- Ablation JSONs are recorded with matching checkpoint/schedule.
- Triplet summary records whether the result is memory-required.

## Log

- 2026-06-02 Not started; normal continuous eval has not passed.

## Review

Status: blocked by normal eval quality, not by missing tooling.
