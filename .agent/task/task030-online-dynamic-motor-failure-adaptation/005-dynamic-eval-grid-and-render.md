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
- 2026-05-21 Extended to fixed `1.8 m/s`. Initial probe from `model_5200.pt`
  failed only on a `max_gravity_xy_after_onset` outlier. A short fixed-speed
  dynamic micro run reached dynamic-switch pass at `model_5239.pt`, but task029
  full regression failed only `dead_motor_grid_07_right_knee_joint`.
- 2026-05-21 Added a right-knee guard rehearsal stage on H200 and resumed from
  `model_5239.pt`. Accepted fixed `1.8 m/s` checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task030_dynamic_004_train/2026-05-21_15-51-26_005_rightknee_vx1p8_from5239_env8192_iter40_gpu1_seed30530/model_5278.pt`.
  Task029 full regression passed, including the previous bottleneck:
  `dead_motor_grid_07_right_knee_joint` with `zero_fall_ratio=0.734375`,
  `lin=0.37535613775253296`, `yaw=0.44515153765678406`. Aggregate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_speed_expansion/eval_model5278_rightknee_vx1p8_task029_regression_full_grid/task029_eval_failure_aggregate.json`.
- 2026-05-21 Fixed `1.8 m/s` dynamic switch multi-seed s5 passed for
  `model_5278.pt`: `pass_count=5/5`, `zero_fall_ratio_min=1.0`,
  `recovery_success_ratio_min=1.0`,
  `post_recovery_lin_vel_error_mean_max=0.1514013409614563`,
  `post_recovery_yaw_vel_error_mean_max=0.2030811458826065`,
  `max_gravity_xy_after_onset_max=0.15313367545604706`. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_speed_expansion/eval_model5278_rightknee_vx1p8_dynamic_switch_s5/task030_dynamic_switch_multiseed_s5_summary.json`.
- 2026-05-21 Rendered fixed `1.8 m/s` dynamic-switch and single-right-knee
  videos for `model_5278.pt`; both cases `pass=true`, `done_count=0`, 500
  frames at 50 FPS. H200 render directory:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_speed_expansion/render_model5278_rightknee_dynamic_vx1p8`.
  Local copies:
  `D:\guoyubo.9\Documents\New project 2\_worktrees\h200-locomotion-lab-task030-online-dynamic-failure\outputs\task030\render_model5278_rightknee_dynamic_vx1p8\`.
- 2026-05-21 Extended to fixed `2.0 m/s`. Probe from the accepted `1.8 m/s`
  checkpoint failed only on the same posture-outlier metric
  (`max_gravity_xy_after_onset=0.952745795249939`) while fall/recovery and
  tracking metrics were already within threshold. A short `2.0 m/s` dynamic
  micro run produced `model_5317.pt`, which passed dynamic-switch eval but
  failed task029 full regression on `dead_motor_grid_07_right_knee_joint`.
- 2026-05-21 A right-knee-only `2.0 m/s` rehearsal fixed right knee but shifted
  the bottleneck to `dead_motor_grid_04_left_hip_roll_joint`. A mixed
  knee+hip-roll guard stage balanced those failure modes and produced the final
  accepted `2.0 m/s` checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task030_dynamic_004_train/2026-05-21_17-35-22_005_kneehiproll_vx2p0_from5320_env8192_iter30_gpu1_seed30750/model_5349.pt`.
- 2026-05-21 Final fixed `2.0 m/s` task029 full regression passed for
  `model_5349.pt`: aggregate `pass=true`, clean pass, motor-primitives pass,
  in-distribution persistent failure pass, doubled holdout pass, and all 12
  forced-dead grid cases pass. Bottleneck grid cases after balancing:
  `dead_motor_grid_04_left_hip_roll_joint zero_fall_ratio=1.0` and
  `dead_motor_grid_07_right_knee_joint zero_fall_ratio=0.6171875`. Aggregate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_speed_expansion/eval_model5349_kneehiproll_vx2p0_task029_regression_full_grid/task029_eval_failure_aggregate.json`.
- 2026-05-21 Final fixed `2.0 m/s` dynamic switch multi-seed s5 passed for
  `model_5349.pt`: `pass_count=5/5`, `zero_fall_ratio_min=1.0`,
  `recovery_success_ratio_min=1.0`,
  `post_recovery_lin_vel_error_mean_max=0.1684277504682541`,
  `post_recovery_yaw_vel_error_mean_max=0.19378596544265747`,
  `max_gravity_xy_after_onset_max=0.14636270701885223`. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_speed_expansion/eval_model5349_kneehiproll_vx2p0_dynamic_switch_s5/task030_dynamic_switch_multiseed_s5_summary.json`.
- 2026-05-21 Rendered final fixed `2.0 m/s` clean, single-right-knee dynamic,
  and switch dynamic videos for `model_5349.pt`; all cases `pass=true`,
  `done_count=0`, 500 frames at 50 FPS. H200 render directory:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_speed_expansion/render_model5349_kneehiproll_dynamic_vx2p0`.
  Local copies:
  `D:\guoyubo.9\Documents\New project 2\_worktrees\h200-locomotion-lab-task030-online-dynamic-failure\outputs\task030\render_model5349_kneehiproll_dynamic_vx2p0\`.
- 2026-05-21 Added a stricter per-joint dynamic single-onset grid at fixed
  `2.0 m/s` for the accepted `model_5349.pt`. This broader route does not pass:
  `pass_count=8/12`; failed dynamic full-dead onset cases are
  `left_hip_pitch_joint`, `left_hip_yaw_joint`, `right_hip_pitch_joint`, and
  `right_knee_joint`. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_speed_expansion/eval_model5349_kneehiproll_vx2p0_dynamic_onset_grid/task030_dynamic_onset_grid_summary.json`.
- 2026-05-21 Tried an all-joint dynamic-onset guard stage from `model_5349.pt`.
  Final `model_5428.pt` did not improve the broader onset grid:
  `pass_count=7/12`; failed cases include hip pitch/yaw and right knee. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_speed_expansion/eval_model5428_alljoint_vx2p0_dynamic_onset_grid/task030_dynamic_onset_grid_summary.json`.
- 2026-05-21 Tried a focused dynamic-onset guard stage from `model_5349.pt` for
  the failed joints. Final `model_5468.pt` also did not close the broader onset
  grid and regressed to `pass_count=6/12`. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_speed_expansion/eval_model5468_focus_vx2p0_dynamic_onset_grid/task030_dynamic_onset_grid_summary.json`.

## Review

Status: partial. The staged ladder closed in order at `1.6 -> 1.8 -> 2.0 m/s`
for the specified task030 dynamic-switch route. The best scoped checkpoint is
`model_5349.pt` from the knee+hip-roll `2.0 m/s` guard run; it passes task029
full regression, dynamic-switch multi-seed s5, and final clean/single/switch
renders with `done_count=0`.

The stricter all-leg per-joint dynamic single-onset grid does not pass. Two
additional MLP-only onset guard attempts (`model_5428.pt`, `model_5468.pt`)
failed to close it and made the grid worse. Do not claim full arbitrary
mid-episode motor-failure robustness from Task030. Treat `model_5349.pt` as the
best scoped dynamic-switch checkpoint and move arbitrary onset adaptation to a
later task with a changed curriculum and likely explicit history/memory.
