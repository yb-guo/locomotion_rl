# 004: Dynamic Failure MLP Train

## Route

Train the same MLP PPO stack on the first dynamic failure distribution.

Initial training mix:

```text
clean episode                         20%
persistent weak/dead episode          20%
dynamic single-failure episode        45%
dynamic two-segment switch episode    15%
```

Training randomization:

- randomized onset time
- randomized duration
- randomized leg motor target
- randomized weak/dead severity
- at most two dynamic segments per episode

Do not change:

- policy architecture
- actor observation dimension
- action dimension/order
- reward stack except for scheduler-compatible bookkeeping if needed
- robot topology, link inertial values, or contact randomization

Entry decision from subtask 003:

- Start from task029 accepted `Fast1p6 model_4700.pt`.
- First train/evaluate fixed `1.6 m/s`; do not advance to `1.8` or `2.0 m/s`
  until dynamic `1.6 m/s` passes.
- Bias the first dynamic distribution toward `left_knee_joint` dead/recovery,
  because isolated dynamic `single-left-knee` failed while isolated dynamic
  `single-right-hip-yaw` passed.
- Preserve the original mix shape, but make the dynamic single-failure bucket
  left-knee-heavy for the first smoke and H200 run.

Pass:

- 64-env smoke proves the dynamic task can train from the selected checkpoint.
- H200 8192-env run produces checkpoints.
- Intermediate checkpoints are screened; final checkpoint is not blindly
  accepted.
- Clean and persistent robustness are not destroyed.

Fail:

- Training uses explicit actor fault labels.
- Training starts before subtask 002/003 evidence exists.
- A slower or stop-walking policy is accepted as dynamic adaptation.

## Log

- 2026-05-21 Opened.
- 2026-05-21 Added H200 dynamic training task
  `Unitree-G1-Gripper-Flat-DynamicMotorFailure-Train-Fast1p6` with actor
  contract unchanged at `104 -> 31`, no explicit actor fault labels, and only
  `actuator_forcerange` as the dynamic randomization field. Contract evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_train_contract/task030_dynamic_failure_contract_summary.json`.
- 2026-05-21 Fixed the randomized dynamic scheduler inactive-row startup bug
  and vectorized per-step actuator force-range application. 64-env smoke then
  passed from the task029 `Fast1p6 model_4700.pt` checkpoint. Smoke log:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_mlp_train/004_dynamic_train_smoke_env64_iter2_gpu1_seed30043.stdout.log`.
- 2026-05-21 Ran the first 8192-env randomized dynamic train from task029
  `model_4700.pt`, then continued from `model_4899.pt`. The best randomized
  checkpoint was `model_5100.pt`, which nearly closed deterministic switch
  (`zero_fall_ratio=0.984375`, `recovery_success_ratio=0.99609375`) but failed
  the strict `max_gravity_xy_after_onset <= 0.8` gate with one outlier
  (`max=0.9426`). JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_mlp_train/eval_model5100_switch_vx1p6/task030_dynamic_eval_switch_vx1p6.json`.
- 2026-05-21 Tested low-entropy and switch-heavy guard continuations. The long
  low-entropy path regressed switch robustness; switch-heavy `model_5125`
  improved but still failed the strict max-gravity gate. These were rejected
  instead of accepting training reward.
- 2026-05-21 Ran canonical dynamic-switch micro fine-tune from `model_5100.pt`.
  `model_5120.pt` and `model_5125.pt` passed deterministic dynamic switch, but
  task029 full-grid regression showed persistent forced-dead grid regressions
  on `left_hip_yaw_joint` and `right_knee_joint`, so they were not accepted.
- 2026-05-21 Added persistent all-critical rehearsal from canonical
  `model_5125.pt`, then continued once from `model_5164.pt`. Accepted checkpoint
  is:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task030_dynamic_004_train/2026-05-21_15-03-09_004_persistent_rehearsal_continue5164_env8192_iter40_gpu1_seed30051/model_5200.pt`.
- 2026-05-21 Accepted `model_5200.pt` for the fixed `1.6 m/s` subtask 004 gate:
  dynamic switch multi-seed s5 passed with `pass_count=5/5`,
  `zero_fall_ratio_min=1.0`, `recovery_success_ratio_min=1.0`,
  `post_recovery_lin_vel_error_mean_max=0.158173069357872`,
  `post_recovery_yaw_vel_error_mean_max=0.21012352406978607`, and
  `max_gravity_xy_after_onset_max=0.18700484931468964`. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_mlp_train/eval_model5200_rehearsal2_dynamic_switch_s5/task030_dynamic_switch_multiseed_s5_summary.json`.
- 2026-05-21 Task029 full regression also passed for the same `model_5200.pt`
  at fixed `1.6 m/s`: aggregate `pass=true`, clean pass, motor-primitives pass,
  in-distribution persistent failure pass, doubled holdout pass, and all 12
  forced-dead grid cases pass. Aggregate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_mlp_train/eval_model5200_rehearsal2_task029_regression_vx1p6_full_grid/task029_eval_failure_aggregate.json`.
- 2026-05-21 Rendered a 10 s deterministic dynamic-switch video for the same
  accepted checkpoint. Case summary `pass=true`, `done_count=0`, 500 frames at
  50 FPS. H200 video:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_mlp_train/render_model5200_dynamic_switch_vx1p6/task030-render-model5200-dynamic-switch-vx1p6-failure_indistribution_forward_0p5.mp4`.
  Local copy:
  `D:\guoyubo.9\Documents\New project 2\_worktrees\h200-locomotion-lab-task030-online-dynamic-failure\outputs\task030\render_model5200_dynamic_switch_vx1p6\task030-render-model5200-dynamic-switch-vx1p6.mp4`.

## Review

Status: passed for fixed `1.6 m/s` dynamic MLP training. The accepted
checkpoint keeps the first-pass MLP actor/action contract (`104 -> 31`),
passes deterministic dynamic switch multi-seed validation, and preserves the
task029 clean/persistent full-grid gate. Later speed expansion to `1.8` and
`2.0 m/s` belongs to subtask 005.
