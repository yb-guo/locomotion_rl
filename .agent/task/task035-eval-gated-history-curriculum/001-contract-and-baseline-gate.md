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
- 2026-05-28 H200 baseline gate initially exposed script/env issues, not policy
  failures:
  - `model5350_baseline_gate`: missing `IPython`;
  - `model5350_baseline_gate_rerun1`: missing `platformdirs` through `wandb`;
  - `model5350_baseline_gate_rerun2`: missing `src.tasks`;
  - early rerun3 dynamic failures used the wrong eval task id.
- 2026-05-28 Fixed runbook script to reuse `/tmp/task029_ipython_stub`, include
  external MJLab on `PYTHONPATH`, and pass the Task034 dynamic task id
  `Unitree-G1-Gripper-Flat-Task033-StackMlpK4-DynamicMotorFailure-Fast1p6`.
- 2026-05-28 Corrected baseline gate passed at `2.0 m/s`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task035/model5350_baseline_gate_rerun3/task035_model5350_baseline_gate_summary.json`.
  Result: dynamic switch `3/3` seeds pass with `zero_fall_ratio=1.0` and
  `recovery_success_ratio=1.0`; full 12-joint dead-grid `3/3` seeds pass
  `12/12`.

## Review

Status: passed for the scoped `2.0 m/s` baseline gate. This does not validate
low-speed dead-grid robustness.
