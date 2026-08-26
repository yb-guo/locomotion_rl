# Task 052: Whole-Body Adapter and Motor Process

## Route

Add explicit 45-slot mappings for generated robots, G1, Berkeley Humanoid,
and ANYmal C.  Add batched contract helpers and the hidden online motor event
process used by later PPO/TXL tasks.

## Log

- 2026-08-19: Added explicit named mappings and a deterministic motor event
  scheduler with weak/dead/latency/recovery semantics.

## Review

Verified: focused contract tests pass for G1 (29 active), Berkeley Humanoid
(12), ANYmal C (12), inactive masks, deterministic event scheduling, and
actor/critic information boundaries.  The masked PPO smoke also confirms
inactive slots do not enter likelihood or entropy.  Full dynamic-fault quality
gates remain in Task055.
