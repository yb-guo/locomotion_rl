# 003 Frozenbase Train

## Route

Continue from Task033 frozen-base `model_5378.pt` using the Task034 focused
stage.

Initial training settings:

- envs: `8192`
- max iterations: first smoke `1`, then focused pass `20-40`
- learning rate: start at `5e-6` or `1e-5`
- entropy coef: keep low, do not encourage gait destruction
- runner: `Task033StackMlpK4FrozenBaseRunner`

## Log

- 2026-05-28 Planned.
- 2026-05-28 Mixed stage continuation from Task033 `model_5378.pt` completed
  30 iterations:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task034_rightknee_frozenbase_train/2026-05-28_14-12-50_034_mixed_rightknee_from_task033_model5378_env8192_iter30_gpu1_seed3403401_lr5e6`.
  Checkpoint bisection over `model_5380`, `5385`, `5390`, `5395`, `5400`,
  `5405`, and `5407` showed all right-knee evals below the baseline. Final
  `model_5407.pt` had `zero_fall_ratio=0.0234375` in the bisection run.
- 2026-05-28 Weak stage continuation from Task033 `model_5378.pt` completed
  10 iterations:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task034_rightknee_frozenbase_train/2026-05-28_14-21-49_034_weak_rightknee_from_task033_model5378_env8192_iter10_gpu1_seed3403411_lr1e6`.
  It also did not improve right-knee; best observed `zero_fall_ratio=0.1171875`.

## Review

Status: failed for the continuation hypothesis. Training right-knee-focused
continuations from `model_5378.pt` regresses the target metric, so the task
should not continue tuning this stage blindly.
