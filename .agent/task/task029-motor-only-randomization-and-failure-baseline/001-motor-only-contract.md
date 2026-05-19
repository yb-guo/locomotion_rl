# 001: Motor-Only Contract

## Route

Freeze the task029 contract before implementation so later failures can be
attributed to motor randomization rather than hidden environment changes.

Allowed first-pass randomization:

- Motor `kp/kd` gain scale.
- Motor effort/strength scale.
- Joint damping and joint friction when used as actuator-side uncertainty.
- Motor torque noise, torque bias, and deadband.
- Action/actuator delay and actuator low-pass/bandwidth after step-response
  validation.
- Episode-start persistent weak/dead leg motor scale.

Forbidden first-pass training randomization:

- Link geometry changes.
- Link mass scale.
- Body COM offset.
- Body inertia or pseudo-inertia randomization.
- Contact friction randomization.
- Terrain changes beyond the existing flat walking setup.
- Encoder bias, observation noise, or actor observation corruption.
- Push disturbances.
- Explicit actor observation of `motor_scale`, `failure_mask`, or fault labels.

Information boundary:

- Actor observation shape and meaning must remain compatible with task028.
- Actor action output must remain the same 31-dim task028 contract.
- Critic may receive privileged motor scale/failure data.
- Eval and render summaries must record true randomization/failure parameters
  for diagnosis, but those values must not enter actor inputs.

## Minimal Closed Loop

Feedback loop:

1. Implement or reuse an inspect script that dumps active events,
   obs/action dimensions, actor/critic obs terms, actuator randomization
   ranges, and failure sampling settings.
2. Assert forbidden events are absent from the training stage.
3. Assert actor obs/action dimensions match task028.
4. Assert any privileged motor fault fields appear only in critic diagnostics
   or critic obs, never actor obs.

Pass:

- Fixed topology and action dimension are preserved.
- Actor observation contract is unchanged from task028.
- No link/contact/sensor randomization is active in the motor-only training
  stage.
- Critic privileged fields, if present, are explicitly listed.
- Inspect output is saved as JSON.

Fail:

- Any forbidden event is active in the first-pass training stage.
- Actor receives motor scale or failure mask.
- Action dimension, action order, or actor observation dimension changes.
- The contract cannot be checked with an agent-runnable script.

Evidence:

- Planned output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/motor_only_contract/`.

## Log

- 2026-05-19 Opened with the user-approved boundary: actor does not see motor
  failure labels, while critic may receive privileged motor information.

## Review

Status: pending.

This subtask is the gate that keeps task029 from becoming a mixed dynamics
randomization task. Do not start training until the motor-only stage can be
inspected and the forbidden randomization list is machine-checked.
