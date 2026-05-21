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

## Review

Status: open. No PPO training should start before this harness has JSON
evidence.
