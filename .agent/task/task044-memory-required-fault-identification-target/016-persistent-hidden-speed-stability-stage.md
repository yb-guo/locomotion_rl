# 016: Persistent Hidden Speed-Stability Stage

## Route

The closest checkpoint so far is the persistent-hidden continuation. It keeps
full-final speed near the quality gate, but still fails on velocity and
stability extremes. The immediate left-knee curriculum made the robot stable
but much slower, so the next repair should stay on the persistent-hidden
distribution and gently improve both speed and posture.

This stage adds:

- `track_linear_velocity.weight=4.0`;
- `track_linear_velocity.std=0.8`;
- `body_orientation_l2.weight=-2.0`;
- `is_terminated.weight=-300.0`;
- unchanged actor-visible observation contract;
- unchanged Task044 triplet eval contract.

## Acceptance

- Local tests lock the new task id and reward changes.
- H200 registry contains the new task id.
- H200 smoke passes.
- H200 continuation train records train-pipeline evidence.
- H200 normal eval records whether final quality improves over the
  persistent-hidden `iter150` checkpoints.
- Full triplet eval is required before any pass claim.

## Log

- 2026-06-01 Added
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenSpeedStability1p6`.

## Review

Status: open.

No H200 evidence yet.
