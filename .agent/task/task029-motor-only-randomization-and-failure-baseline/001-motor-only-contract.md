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
- 2026-05-19 H200 inspect passed. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/motor_only_contract/summary.json`.
  Six MotorOnly tasks passed the contract checks with `action_dim=31`,
  `actor_obs=104`, and `critic_obs=119`.
- 2026-05-19 Inspect confirmed actor corruption is false and forbidden
  link/contact/sensor/push events are absent. Temporary `PYTHONPATH` stubs were
  used only for missing inspect-time dependencies and did not modify the conda
  environment.

## Review

Status: passed.

The motor-only contract is machine-checked for the first acceptance stage.
Task028-compatible action and actor-observation dimensions are preserved, the
critic has the expected larger privileged observation shape, and no forbidden
link/contact/sensor/push randomization was active in the inspected MotorOnly
tasks.
