# Task 058: Scale and Independent Flow Algorithms

## Route

Only after Task057 gates pass, scale structural/physical instances and add
terrain/pushes.  Keep MIP/JiT/flow matching independent of the task schema:
connect to PPO only when a true tractable log probability exists; otherwise
use advantage/Q-weighted flow matching.

## Log

- 2026-08-19: Added guarded flow-matching PPO and advantage/Q-weighted
  regression adapters without changing the 45D schema or split protocol.

## Review

Blocked on Task057 evidence by design; no scale-up claim is made yet.  The
flow adapter rejects missing likelihoods for PPO and exposes the
advantage/Q-weighted fallback.
