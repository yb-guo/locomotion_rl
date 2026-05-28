# 006 Bakeoff Eval Decision

## Route

Compare the three memory consumers using shared eval criteria.

Required first eval:

- blocker subset:
  - speeds `0.4`, `1.6`, `2.0 m/s`;
  - forced persistent dead-grid;
  - canonical dynamic switch.

Escalate to full eval only for consumers that improve the subset:

- speeds `0.4`, `0.8`, `1.2`, `1.6`, `2.0 m/s`;
- clean;
- random persistent motor failure;
- forced persistent 12-joint dead-grid;
- canonical dynamic switch;
- Level C arbitrary onset diagnostic only if blocker subset improves.

Decision outputs:

- overhead table;
- eval table;
- best checkpoint per consumer;
- recommendation for next task.

## Log

- 2026-05-28 Planned as final Task033 review.
- 2026-05-28 Initial overhead smokes exist for baseline MLP, buffer-only K4,
  StackMLP K4/K8, GRU K4, and token K4. These are construction/cost smokes,
  not final throughput benchmarks.
- 2026-05-28 StackMLP K4 was selected for the first policy-quality eval because
  it has a direct `model_5349` migration route. The migrated checkpoint is
  behavior-compatible with the base MLP on dynamic switch, but naive PPO on the
  expanded input destroys gait quality.
- 2026-05-28 Frozen-base StackMLP K4 is the current best Task033 result. It
  passes `2.0 m/s` dynamic switch and improves `2.0 m/s` forced dead-grid from
  Task032's `8/12` to `11/12`, but still fails `right_knee_joint` forced dead.
  Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/stackmlp_k4_eval/frozenbase_model5378_dynamicmotorfailure_vx2p0_seed3105349/task033_dynamic_eval_switch_vx2p0.json`;
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/stackmlp_k4_eval/frozenbase_model5378_deadgrid_vx2p0_seed3303500/task033_failure_grid_eval_aggregate.json`.
- 2026-05-28 Task034 follow-up checkpoint sweep superseded the Task033
  `model_5378.pt` selection for deployment: earlier checkpoint `model_5350.pt`
  from the same frozen-base run passes both `2.0 m/s` dynamic switch and full
  `2.0 m/s` forced dead-grid. Task033 remains the infrastructure bakeoff
  record; Task034 owns the corrected right-knee decision.

## Review

Status: decision-ready close. Shared history infra is viable, GRU and token
consumers are wired cheaply enough to keep, and Task033's policy-quality winner
is frozen-base StackMLP K4. Task033 originally stopped at `model_5378.pt`
(`11/12` full dead-grid); Task034 then selected `model_5350.pt` from the same
run as the corrected robust checkpoint.
