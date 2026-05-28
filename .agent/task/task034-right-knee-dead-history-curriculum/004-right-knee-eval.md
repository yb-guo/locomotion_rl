# 004 Right-Knee Eval

## Route

Evaluate the target case first before spending time on the full matrix.

First gate:

- task: Task034/Task033 frozen-base focused dead-grid eval task;
- checkpoint: latest Task034 focused checkpoint;
- case: `right_knee_joint` forced dead;
- speed: `2.0 m/s`;
- envs/steps: match Task033 baseline, `128 envs x 500 steps`.

If right-knee passes or materially improves, run full 12-joint dead-grid at
`2.0 m/s`.

## Log

- 2026-05-28 Planned.
- 2026-05-28 Checkpoint bisection over the original Task033 frozen-base run
  found `model_5350.pt` passes the right-knee target, while later checkpoints
  regress it:
  - `model_5350`: right-knee pass;
  - `model_5360`: `zero_fall_ratio=0.3125`;
  - `model_5370`: `zero_fall_ratio=0.25`;
  - `model_5378`: `zero_fall_ratio=0.1953125`.
- 2026-05-28 Full `2.0 m/s` 12-joint dead-grid eval for Task033 frozen-base
  `model_5350.pt` passed `12/12`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task034/frozenbase_model5350_deadgrid_vx2p0_seed3403500/task033_failure_grid_eval_aggregate.json`.

## Review

Status: passed. `model_5350.pt` closes the right-knee target and full
`2.0 m/s` forced dead-grid.
