# 002: Dynamic Failure Scheduler Harness

## Route

Implement a deterministic mid-episode scheduler and trace it before training.

Required eval template:

```text
0.0s - 2.0s    normal
2.0s - 4.0s    left_knee weak/dead
4.0s - 5.0s    normal recovery
5.0s - 7.0s    right_hip_yaw weak/dead
7.0s - end     normal recovery
```

The harness must record, per policy step:

- policy step index and time in seconds
- active segment id
- active joint name
- failure type
- force/torque scale
- whether the step is inside the `0.3 s` transient window
- root/base state metrics used by dynamic eval

Pass:

- Trace shows normal, fault, recovery, and switch segments at expected times.
- Active fault state changes mid-episode without env reset.
- Fault state is restored after recovery segments.
- Actor observation terms remain unchanged.

Fail:

- Dynamic failure is implemented as reset-time failure.
- Force ranges/scales accumulate across segment changes.
- Scheduler state leaks into actor observations.

## Log

- 2026-05-21 Opened.
- 2026-05-21 Implemented the deterministic `step` scheduler template in the
  H200 MJLab G1 gripper velocity config:
  `normal -> left_knee_joint dead -> normal -> right_hip_yaw_joint dead -> normal`.
- 2026-05-21 Ran scheduler trace with task029 accepted checkpoint
  `model_4700.pt`, fixed `lin_vel_x=1.2`, `num_envs=16`, `steps=420`,
  device `cuda:0`. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_scheduler_trace/task030_dynamic_scheduler_trace_summary.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task030/dynamic_failure_scheduler_trace/task030_dynamic_scheduler_trace.json`.
- 2026-05-21 Trace passed: `pass=true`, `trace_pass=true`,
  `done_count_env0=0`, `transition_count=5`. Observed transitions:
  - `0.02s`: segment 0, normal.
  - `2.00s`: segment 1, `left_knee_joint`, dead, scale `0.0`,
    `transient=true`.
  - `4.00s`: segment 2, normal recovery, scale `1.0`,
    `transient=true`.
  - `5.00s`: segment 3, `right_hip_yaw_joint`, dead, scale `0.0`,
    `transient=true`.
  - `7.00s`: segment 4, normal recovery, scale `1.0`,
    `transient=true`.

## Review

Status: passed. The scheduler changes active failure state mid-episode without
reset, restores normal scale on recovery segments, and records segment/fault
state in JSON artifacts without changing actor observations.
