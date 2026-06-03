# 018: Speed-Pose Balance Stage

## Route

Pose-tight training moved the hidden left-knee fault eval close to the strict
normal gate: fall, root height, and gravity could pass, but final
`lin_vel_error.mean` stayed around `0.49-0.50`. A short speed nudge improved
velocity only slightly and reintroduced pose drift.

This subtask adds one narrow stage,
`PersistentHiddenSpeedPoseBalance1p6`, that keeps the same hidden-fault
schedule, actor-visible observation contract, action shape, runner, and eval
gate. It only changes train-time reward/termination shaping:

- `track_linear_velocity.weight=6.5`
- `track_linear_velocity.std=0.55`
- `body_orientation_l2.weight=-4.0`
- `is_terminated.weight=-400.0`
- `gravity_xy_too_high.max_xy=0.72`

## Acceptance

- Local tests lock the new task id, helper, registration, and train/eval
  allowlists.
- H200 registry patch writes the new helper and task id.
- H200 smoke must pass before continuation training.
- H200 normal eval must be recorded with strict Task044 quality feedback.
- No Task044 pass claim is allowed unless normal quality passes and the full
  normal / zero-residual / stateless triplet proves memory-required behavior.

## Log

- 2026-06-01 Created after pose-tight continuation failed to close the speed
  gap. Current best normal eval before this subtask was:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_pose_tight_lr1e5_from_pose_iter10_model9_normal_probe_left_knee_joint_vx1p6_seed4411401.json`
  with `fall_ratio=0.0078125`, `gravity_xy.max=0.7357999682426453`,
  `root_z.min=0.5745701193809509`, and
  `lin_vel_error.mean=0.4907352030277252`.
- 2026-06-01 Short continuation from that checkpoint did not pass. LR1e-5
  continuation:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_pose_tight_lr1e5_cont_from_tight_iter5_model4_normal_probe_left_knee_joint_vx1p6_seed4411701.json`
  failed with `lin_vel_error.mean=0.5035483837127686`.
  LR2e-5 continuation:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_pose_tight_lr2e5_cont_from_tight_iter5_model4_normal_probe_left_knee_joint_vx1p6_seed4411801.json`
  failed with `gravity_xy.max=0.7584195137023926` and
  `lin_vel_error.mean=0.4884888529777527`.

## Review

Status: active, not passed.

The remaining failure is a velocity/posture tradeoff, not missing files or a
stuck H200 job. This stage is intentionally small so the next evidence can
answer whether stricter pose termination plus stronger speed reward is enough
before changing policy architecture or relaxing the gate.
