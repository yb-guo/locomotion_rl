# 002: Stability Tail Objective Stage

## Route

The next stage should target the residual post-fault fall/reset tail without
relaxing eval or changing actor-visible observations. Candidate routes must be
judged by whether they reduce fall ratio while preserving the already-good
speed and posture.

## Acceptance

- Select or add one narrow training objective.
- Keep G1-like topology, action shape, visible observation contract, and
  continuous eval thresholds unchanged.
- H200 smoke or direct train evidence must exist before calling the stage
  usable.

## Log

- 2026-06-02 Opened after repeated-seed audit showed stable failure at
  `fall_ratio=0.105-0.125`.
- 2026-06-02 Selected the narrow survival objective recommended by read-only
  review: add
  `Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneePoseForwardSurvival1p6`.
  It inherits the current best fixed left-knee pose-forward stage and changes
  only survival/posture pressure:
  - `body_orientation_l2.weight=-5.0`;
  - `is_terminated.weight=-700.0`;
  - `gravity_xy_too_high.max_xy=0.70`.
  It intentionally does not add new speed pressure because current best evals
  already satisfy `lin_vel_error <= 0.45`.
- 2026-06-02 Added a long-survival companion task after inspecting the
  continuous eval contract. The eval requires `physical_reset_events == 0`
  across a 360-step continuous rollout, while the fixed left-knee curriculum
  was only a 2.0 s episode. The companion task
  `Unitree-G1-Gripper-Flat-Task045-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeLongSurvival1p6`
  inherits the survival objective and sets `episode_length_s=8.0` with
  `dynamic_single_duration_range_s=(8.0, 8.0)` so training covers the same
  long-horizon no-reset behavior the eval requires.

## Review

Status: implementation in progress, not passed.

The route is selected and local tests are pending. H200 evidence is still
required before this subtask can pass.
