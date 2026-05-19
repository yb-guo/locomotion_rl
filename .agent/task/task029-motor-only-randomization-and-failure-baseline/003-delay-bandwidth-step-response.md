# 003: Delay Bandwidth Step Response

## Route

Implement and validate action/actuator delay plus actuator low-pass or
bandwidth limits. This subtask uses a step-response harness before PPO because
delay bugs can produce plausible training runs with incorrect timing.

Candidate mechanisms:

- Action delay in policy-step units, initially `0-2` steps.
- Actuator delay if supported by the MJLab actuator wrapper.
- Low-pass filtering or bandwidth limits on target action or realized torque.
- Optional wider holdout delay after the base delay stage is verified.

## Minimal Closed Loop

Feedback loop:

1. Build a deterministic one-env step-response script.
2. Apply a known action step to selected joints.
3. Record policy action, delayed/filtered action, realized target, torque,
   joint position, and joint velocity over time.
4. Run baseline, delay-only, low-pass-only, and delay+low-pass settings.
5. Save plots or JSON summaries that make timing differences visible.

Pass:

- Delay of `N` policy steps shifts the applied target/torque by `N` steps.
- Low-pass/bandwidth setting smooths the applied target/torque without changing
  the raw policy action.
- The implementation preserves task028 actor action and observation contracts.
- The validation does not depend on PPO learning.

Fail:

- Delay is implemented but not observable in the step-response trace.
- Filtering mutates the logged raw actor action instead of the actuator input.
- PPO is used as the first evidence for delay correctness.
- Delay/bandwidth changes topology, action dim, or actor obs dim.

Evidence:

- Planned output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/delay_bandwidth_step_response/`.

## Log

- 2026-05-19 Opened because task028 deferred delay/smoothing after finding it
  likely requires actuator-wrapper work rather than a simple event toggle.

## Review

Status: pending.

This is a timing-boundary diagnostic subtask. It should finish with a fast,
deterministic harness that can catch off-by-one delay and incorrect filter
placement before any long H200 training run starts.
