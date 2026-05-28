# 006 Decision

## Route

Make a narrow decision after one focused frozen-base pass.

Decision options:

- `frozen-base enough`: right-knee and full `2.0 m/s` dead-grid pass while
  dynamic switch remains passed.
- `frozen-base promising`: right-knee materially improves but does not pass.
- `frozen-base ceiling`: right-knee does not improve or dynamic switch regresses.

If frozen-base ceiling is reached, the next task should move to GRU/token
long-training or explicit adaptation, not keep tuning the same stage blindly.

## Log

- 2026-05-28 Planned.
- 2026-05-28 Decision: `frozen-base enough` for the narrow `2.0 m/s`
  right-knee/full-dead-grid target, but not through new right-knee-focused
  training. The accepted checkpoint is the earlier Task033 frozen-base
  `model_5350.pt`.

## Review

Status: passed. Use:
`/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task033_stackmlp_k4_frozenbase_focused/2026-05-28_12-40-56_033_frozenbase_focused_from5349_env8192_iter30_gpu1_seed3303362_lr1e5/model_5350.pt`.

Do not use the Task034 mixed/weak continuation checkpoints for deployment or
further training; they regress the target metric.
