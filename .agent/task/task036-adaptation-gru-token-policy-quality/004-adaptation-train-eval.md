# 004 Adaptation Train Eval

## Route

Train AdaptK4 from the base MLP warmstart and evaluate by checkpoint sweep.

First gates:

- `2.0 m/s` dynamic switch;
- `2.0 m/s` right-knee/right-hip-pitch dead fast gate;
- `0.4 m/s` left-hip-yaw/left-hip-roll/right-knee dead fast gate.

Escalate to full matrix only for candidates that pass these fast gates.

## Log

- 2026-05-28 H200 warmstart train completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task036/policy_train/036_adapt_k4_warmstart_env8192_iter60_gpu0_seed3603630.stdout.log`.
  The run resumed from base `model_5349` and saved final checkpoint
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task036_policy_quality_train/2026-05-28_23-10-53_036_adapt_k4_warmstart_env8192_iter60_gpu0_seed3603630/model_5408.pt`.
- 2026-05-28 Full matrix eval for `model_5408` completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task036/full_validation_adapt_adapt_model5408_iter60/task036_full_validation_summary.json`.
  Result: `pass=false`. Dynamic switch passed at `0.4/1.2/2.0`.
  Deadgrid passed `9/12` at `0.4`, `12/12` at `1.2`, and `11/12` at
  `2.0`; failures were `left_hip_yaw_joint`, `left_hip_roll_joint`, and
  `right_knee_joint` at `0.4`, plus `right_knee_joint` at `2.0`.
- 2026-05-28 Focused checkpoint sweep over `model_5398/5400/5402/5404/5406/5408`
  completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task036/adapt_focused_sweep/task036_adapt_focused_sweep_summary.json`.
  No checkpoint passed the focused failure set.

## Review

Status: partial. AdaptK4 is much closer than GRU/token 60-iter scratch, but it
is not a promoted checkpoint because the full matrix did not pass.
