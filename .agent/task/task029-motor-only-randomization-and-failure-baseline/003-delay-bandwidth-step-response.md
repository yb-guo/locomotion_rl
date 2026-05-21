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
- 2026-05-19 H200 step-response validation passed. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/delay_bandwidth_step_response/summary.json`.
  The harness preserved `action_dim=31` and `actor_obs=104`.
- 2026-05-19 Step-response assertions passed: baseline raw action remained
  unmutated and had no delay; delay-only shifted the applied target; low-pass
  had no delay and smoothed the first step; delay+low-pass shifted the smoothed
  applied target.
- 2026-05-19 Key timing values: `step_at=3`, `delay_steps=2`,
  `low_pass_alpha=0.35`, target joint `left_knee_joint`,
  `action_scale` approximately `0.350661`, baseline `target_delta` at step 3
  approximately `0.350661`, low-pass step 3 approximately `0.122731`, and
  delay-only applied at step 5.

## Review

Status: passed.

The deterministic harness verifies the timing boundary before PPO training.
Delay placement, low-pass placement, and raw-action logging behavior are covered
by saved JSON evidence, including the expected two-step shift and first-step
low-pass smoothing magnitude.
