# Task 057: Frozen Heldout and OOD Evaluation

## Route

Freeze one checkpoint and run procedural heldout topology, doubled dynamics,
dynamic motor, locked/stuck motor, Berkeley Humanoid, ANYmal C, G1, and Go2
regression cases.  Named robots may only provide mappings and nominal limits;
they never enter training or checkpoint selection.

## Log

- 2026-08-19: Added the frozen OOD case list, named-robot mapping hooks,
  threshold object, suite runner, and paired bootstrap CI helper.

## Review

The frozen case list and gate/CI helpers are implemented and named mappings
compile (including Go2's explicit 12-slot map).  Actual zero-shot/few-shot
rollout measurements remain pending and no OOD claim is made.
