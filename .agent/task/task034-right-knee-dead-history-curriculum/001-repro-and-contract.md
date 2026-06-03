# 001 Repro And Contract

## Route

Lock the failure before tuning.

Baseline evidence from Task033:

`/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/stackmlp_k4_eval/frozenbase_model5378_deadgrid_vx2p0_seed3303500/task033_failure_grid_eval_aggregate.json`

Failure:

- case: `dead_motor_grid_07_right_knee_joint`
- `zero_fall_ratio=0.2109375`
- `lin_vel_error_mean=0.5108156800270081`
- `yaw_vel_error_mean=0.522580087184906`
- `gravity_xy_mean=0.08332604914903641`

Acceptance threshold is unchanged:

- `zero_fall_ratio >= 0.50`
- `lin_vel_error_mean <= 1.00`
- `yaw_vel_error_mean <= 1.00`
- `gravity_xy_mean <= 0.75`

## Log

- 2026-05-28 Baseline failure recorded from Task033 evidence.
- 2026-05-28 Same-seed baseline for Task033 `model_5378.pt` at seed
  `3403500` reproduced the target failure:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task034/rightknee_eval/task033_model5378_vx2p0_seed3403500/task033_failure_grid_eval_aggregate.json`
  (`zero_fall_ratio=0.1796875`).

## Review

Status: passed. The target failure is reproducible and seed-sensitive but
consistently below threshold.
