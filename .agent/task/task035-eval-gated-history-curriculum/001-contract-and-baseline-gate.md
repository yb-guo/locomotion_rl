# 001 Contract And Baseline Gate

## Route

Freeze the validation and curriculum contract before running more training.

Candidate baseline:

`/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_stackmlp_k4_frozenbase_focused/2026-05-28_12-40-56_033_frozenbase_focused_from5349_env8192_iter30_gpu1_seed3303362_lr1e5/model_5350.pt`

Regression baseline:

`/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_stackmlp_k4_frozenbase_focused/2026-05-28_12-40-56_033_frozenbase_focused_from5349_env8192_iter30_gpu1_seed3303362_lr1e5/model_5378.pt`

Baseline gate:

- dynamic switch at `2.0 m/s`, seeds `3503500`, `3503501`, `3503502`;
- full 12-joint forced dead-grid at `2.0 m/s`, seeds `3503500`,
  `3503501`, `3503502`.

Representative curriculum validation speeds:

- `0.4 m/s`;
- `1.2 m/s`;
- `2.0 m/s`.

Inherited thresholds:

- forced dead-grid:
  - `zero_fall_ratio >= 0.50`;
  - `lin_vel_error.mean <= 1.00`;
  - `yaw_vel_error.mean <= 1.00`;
  - `gravity_xy.mean <= 0.75`;
- dynamic switch:
  - `zero_fall_ratio >= 0.90`;
  - `recovery_success_ratio >= 0.75`;
  - `post_recovery_lin_vel_error_mean <= 0.80`;
  - `post_recovery_yaw_vel_error_mean <= 0.80`;
  - `max_gravity_xy_after_onset <= 0.80`.

## Log

- 2026-05-28 Created.

## Review

Status: ready. This subtask is complete when the baseline gate has JSON
evidence, not when the task document exists.
