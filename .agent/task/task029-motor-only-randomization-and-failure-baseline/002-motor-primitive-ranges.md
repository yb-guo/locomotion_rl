# 002: Motor Primitive Ranges

## Route

Add motor-side randomization primitives one group at a time. Each primitive
must be independently toggleable so a training regression can be attributed to
one cause.

Initial primitive stages:

1. `MotorKpKd`
   - Expand `kp/kd` scale from the task028 smoke range toward staged
     robustness, starting near `0.75-1.25` and only widening after smoke.

2. `MotorStrength`
   - Randomize effort/torque strength per actuator or per actuator group.
   - Keep the first range conservative, then stage toward wider holdout ranges.

3. `MotorDampingFriction`
   - Randomize damping, viscous friction, Coulomb friction, and stiction where
     MJLab exposes reliable knobs.

4. `MotorTorqueNoiseBias`
   - Add torque-side noise and bias after confirming where torque is applied in
     the actuator path.

5. `MotorDeadband`
   - Add small command/torque deadband only after the torque path is
     instrumented.

## Minimal Closed Loop

Feedback loop for each primitive:

1. Register a stage-specific task ID.
2. Run inspect and save JSON with active randomizers and ranges.
3. Run 64-env, 2-iteration PPO smoke on H200.
4. Save checkpoint path and command in this subtask log.
5. Confirm no residual training process remains.

Pass:

- Each primitive stage imports, resets, and trains for the short PPO smoke.
- Each primitive preserves action dim 31 and task028 actor obs shape.
- No primitive enables link/contact/sensor randomization.
- Ranges are recorded in inspect JSON.

Fail:

- Multiple primitives are introduced without independent stage IDs.
- A primitive changes topology, action order, action dim, or actor obs dim.
- A smoke failure cannot be attributed to a specific primitive.

Evidence:

- Planned output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/motor_primitives/`.

## Log

- 2026-05-19 Opened from the task029 plan. The task028 `MotorPd` stage is a
  starting point only; task029 must split and strengthen motor primitives
  instead of using one broad bundle.
- 2026-05-19 H200 smoke passed. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/motor_primitives/motor_primitive_smoke_summary.json`.
  Six primitive stages ran with 64 envs for 2 PPO iterations on `gpu0`:
  `control`, `kp_kd`, `strength`, `damping_friction`, `armature`, and
  `combined`.
- 2026-05-19 Each primitive smoke produced `model_1.pt`, agent/env YAML, and a
  TensorBoard event file. `upload-model` was false, and no residual training
  process remained after the runs.
- 2026-05-19 Torque noise, torque bias, and deadband were not included in this
  pass because MJLab does not expose ready-made reliable APIs for those knobs.
  They require a later actuator-wrapper implementation and should not be
  treated as covered by this subtask's H200 pass.

## Review

Status: passed.

The available MJLab motor-side primitives are implemented as independently
toggleable stages and are smoke-runnable before being combined. This pass covers
control, `kp/kd`, strength, damping/friction, armature, and combined motor-side
randomization only. Torque noise, torque bias, and deadband remain deferred to
an actuator-wrapper slice.
