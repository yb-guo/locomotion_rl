# 006 Comparison Decision

## Route

Compare:

- StackMLP scoped baseline;
- AdaptK4;
- GRU K4;
- Token K4.

The decision must state:

- promoted checkpoint, if any;
- partial candidate, if any;
- failed speeds/joints;
- whether the next route should be longer memory, explicit privileged teacher,
  or different training objective.

## Log

- 2026-05-28 Preliminary comparison after first full matrix:
  AdaptK4 `model_5408` is partial, token `model_59` and GRU `model_59` are
  rejected at 60 iterations. GRU/token are continuing to approximately 300
  total iterations before final rejection or escalation.
- 2026-05-28 Final initial bakeoff decision:
  - StackMLP scoped baseline remains the known reference but was not superseded.
  - AdaptK4 `model_5408` is best partial: all dynamic-switch gates passed,
    but full deadgrid failed at `0.4` and `2.0`.
  - GRU K4 `model_298` is rejected for this route: high-speed gates failed.
  - Token K4 `model_298` is rejected for this route: high-speed gates failed.
  - Next route should not be just a consumer swap; it needs an objective or
    curriculum change that prevents stand-still/high-pose local optima and
    specifically trains the remaining dead-motor cases.

## Review

Status: complete. No promoted checkpoint.
