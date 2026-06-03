# 019: Forward Speed Floor Stage

## Route

The new velocity-component eval showed the remaining strict failure is forward
under-speed, not lateral drift. For the current best pose-tight checkpoint, the
final trial had:

- command `vx=1.6`
- actual `vx=1.268424153327942`
- x-direction mean absolute error `0.46766161918640137`
- y-direction mean absolute error `0.07159683853387833`

This subtask adds `PersistentHiddenForwardFloor1p6`, a narrow train-only stage
that keeps the same hidden-fault schedule, runner, actor observation contract,
action shape, and eval gate. It adds one reward term:

- `forward_velocity_below_l1(target_x=1.45, weight=-3.0)`

The stage starts from pose-tight rather than speed-pose-balance, because the
speed-pose-balance continuation did not improve the normal eval trend.

## Acceptance

- Eval summaries include velocity component diagnostics without changing the
  quality gate.
- Per-trial `completion_ratio` is capped at `1.0` so repeated trial completions
  during a fixed-step eval do not create impossible ratios above 100%.
- Local tests lock the new reward patch, task id, helper, train/eval allowlists,
  velocity component metrics, and completion-ratio cap.
- H200 registry patch writes the new reward and task id.
- H200 smoke must pass before continuation training.
- H200 normal eval must be recorded with strict Task044 quality feedback and
  velocity components.
- No Task044 pass claim is allowed unless normal quality passes and the full
  normal / zero-residual / stateless triplet proves memory-required behavior.

## Log

- 2026-06-01 Local validation passed:
  `18 passed, 15 skipped` for Task037/Task041/Task044 targeted tests, and
  `inspect_agent` passed.
- 2026-06-01 H200 smoke passed for `PersistentHiddenForwardFloor1p6`.
- 2026-06-01 H200 continuations from the pose-tight checkpoint did not close
  the original 2.0 s strict gate:
  - LR5e-6 / 10 iters eval:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_forward_floor_lr5e6_from_pose_iter10_model9_normal_probe_left_knee_joint_vx1p6_seed4412501.json`
    with `lin_vel_error.mean=0.5001372694969177`.
  - LR1e-5 / 10 iters eval:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_forward_floor_lr1e5_from_pose_iter10_model9_normal_probe_left_knee_joint_vx1p6_seed4412701.json`
    with `lin_vel_error.mean=0.48872387409210205`.
- 2026-06-01 Related forward-target and speed-curriculum probes also stayed
  near the same failure band instead of materially improving the strict 2.0 s
  final-trial mean. The best continuation remains the original pose-tight
  checkpoint, not the forward-floor stage.

## Review

Status: closed as a failed diagnostic, not passed.

This targeted repair did not train out the measured full-trial under-speed.
The evidence now points to a reset/startup boundary rather than an ordinary
forward reward-weight issue: tail speed can be good, while the full 2.0 s
strict metric remains dominated by the first post-reset acceleration phase.
