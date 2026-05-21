# 001: Dynamic Failure Contract

## Route

Define the exact contract before adding code.

Scope:

- Keep current MLP policy and current actor observation/action contract.
- Actor input remains `104` dims and action remains `31` dims.
- Actor must not observe active fault state, motor scale, failure mask, or
  segment id.
- Critic/eval/logging may record fault state.
- First dynamic fault types are `weak motor` and `dead motor`.
- First dynamic faults target leg motors only.
- First pass excludes locked joints, stuck commands, multi-motor dynamic
  failures, online delay jumps, and contact/link/sensor randomization changes.

Dynamic timing contract:

- Eval starts with deterministic templates.
- Training uses randomized onset, duration, joint, and severity.
- Fault onset/switch allows `0.3 s` transient recovery for tracking metrics.
- Falls, base instability, and severe height loss are still counted during the
  transient window.

## Log

- 2026-05-21 Opened with the user-approved Task030 decisions from planning.
- 2026-05-21 Added task030 artifacts:
  - `artifacts/task030_create_dynamic_failure_stage.py`
  - `artifacts/task030_inspect_dynamic_failure_contract.py`
  - `artifacts/task030_dynamic_scheduler_trace.py`
  - `artifacts/task030_dynamic_eval_checkpoint.py`
- 2026-05-21 Applied the dynamic failure stage to the H200 MJLab checkout. New
  task id:
  `Unitree-G1-Gripper-Flat-DynamicMotorFailure-Fast1p6`.
- 2026-05-21 H200 contract inspect passed. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_contract/task030_dynamic_failure_contract_summary.json`.
  Key fields: `pass=true`, `actor_obs_dim=104`, `action_dim=31`,
  `critic_obs_dim=119`, `available_event_modes=["reset", "step"]`,
  `dynamic_motor_failure.mode="step"`, no reset-time `motor_failure` event, and
  `forbidden_actor_terms=[]`.

## Review

Status: passed. The inspect artifact proves the MLP actor/action contract is
unchanged and no explicit failure/fault/scale/segment term leaks into actor
observations.
