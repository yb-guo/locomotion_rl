# 005: Dynamic Eval Grid And Render

## Route

Evaluate trained dynamic MLP checkpoints and expand speed to `2.0 m/s`.

Per-speed gates:

- fixed command clean eval
- persistent failure eval
- dynamic single-failure eval
- dynamic switch eval
- per-joint dynamic onset grid
- switch grid
- render clean, dynamic single, and dynamic switch videos

Speed ladder:

1. Close `1.6 m/s`.
2. Extend to `1.8 m/s`.
3. Extend to `2.0 m/s`.

Dynamic pass thresholds:

- `zero_fall_ratio >= 0.90`
- `recovery_success_ratio >= 0.75`
- post-recovery `lin_vel_error_mean <= 0.8`
- post-recovery `yaw_vel_error_mean <= 0.8`
- `max_gravity_xy_after_onset <= 0.8`

Pass:

- Every speed stage has JSON summaries and video evidence.
- Render review does not show stop-walking, excessive shaking, dragging, or
  upper-body flailing as the adaptation mechanism.
- `2.0 m/s` is accepted only after `1.6` and `1.8` are closed.

Fail:

- The speed ladder skips directly to `2.0 m/s`.
- Dynamic switch cases are omitted.
- Videos are missing for the final accepted checkpoint.

## Log

- 2026-05-21 Opened.
- 2026-05-21 Subtask 004 produced the first fixed `1.6 m/s` accepted
  checkpoint for this eval/render stage:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task030_dynamic_004_train/2026-05-21_15-03-09_004_persistent_rehearsal_continue5164_env8192_iter40_gpu1_seed30051/model_5200.pt`.
  Dynamic switch multi-seed s5 and task029 full clean/persistent regression
  both pass for that checkpoint.
- 2026-05-21 Rendered one fixed `1.6 m/s` deterministic dynamic-switch case for
  `model_5200.pt`; case `pass=true`, `done_count=0`, 500 frames at 50 FPS.
  Video:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_mlp_train/render_model5200_dynamic_switch_vx1p6/task030-render-model5200-dynamic-switch-vx1p6-failure_indistribution_forward_0p5.mp4`.
  Local copy:
  `D:\guoyubo.9\Documents\New project 2\_worktrees\h200-locomotion-lab-task030-online-dynamic-failure\outputs\task030\render_model5200_dynamic_switch_vx1p6\task030-render-model5200-dynamic-switch-vx1p6.mp4`.

## Review

Status: open. Fixed `1.6 m/s` has an accepted checkpoint and one dynamic-switch
render from subtask 004, but this subtask still needs full dynamic single/grid
coverage and staged `1.8 -> 2.0 m/s` evidence.
