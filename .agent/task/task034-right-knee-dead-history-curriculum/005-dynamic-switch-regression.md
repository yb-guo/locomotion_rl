# 005 Dynamic-Switch Regression

## Route

Preserve the Task033/Task030 dynamic-switch behavior.

Gate:

- speed: `2.0 m/s`;
- canonical switch;
- thresholds unchanged from Task033 dynamic eval.

The task cannot pass if this regresses, even if right-knee improves.

## Log

- 2026-05-28 Planned.
- 2026-05-28 Task033 frozen-base `model_5350.pt` passes `2.0 m/s` canonical
  dynamic switch:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task034/frozenbase_model5350_dynamicmotorfailure_vx2p0_seed3105349/task033_dynamic_eval_switch_vx2p0.json`
  (`pass=true`, `zero_fall_ratio=1.0`, `recovery_success_ratio=1.0`,
  `max_gravity_xy_after_onset=0.15834704041481018`).

## Review

Status: passed.
