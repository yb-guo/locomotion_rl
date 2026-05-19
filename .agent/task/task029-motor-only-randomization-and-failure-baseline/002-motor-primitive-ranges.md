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

## Review

Status: pending.

This subtask should not prove final robustness. It should prove that each
motor-side primitive is implemented, inspectable, independently toggleable, and
smoke-runnable before being combined.
