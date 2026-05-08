# Route

Task: task007-sonic-g1-deployment-bridge

Goal: document the later real G1 low-level state/command adapter plan after the
dry-run backend contract is stable.

Pass condition:

- The real-robot path identifies required LowState fields, command outputs,
  order mappings, safety gates, rates, and dry-run stages.
- The plan explicitly separates body 29DoF locomotion from dexterous hand
  control.

# Log

- 2026-05-08: Backend dry-run evidence passed on H200. The later real G1
  adapter should be implemented as a new route after this plan is reviewed.

## Planned Real G1 Adapter Boundary

The real robot backend should implement the same `G1RobotBackend` contract:

```text
reset/read_state:
  LowState motor q/dq + IMU quaternion/angular velocity
  -> G1RobotState(root_qpos, motor_positions_mujoco, motor_velocities_mujoco)

write_command:
  G1MotorCommand.motor_position_targets_mujoco
  -> LowCmd motor position targets, kp/kv/force limits, mode bits

advance:
  wait for next 50 Hz policy tick / newest LowState
```

Required input fields:

- 29 body motor positions;
- 29 body motor velocities;
- base quaternion in the convention required by SONIC `qw qx qy qz`;
- base angular velocity;
- monotonic timestamp or sequence number;
- robot mode / estop / fault state.

Required output fields:

- 29 body motor position targets in MuJoCo/hardware order;
- zero or policy-defined velocity targets;
- official SONIC G1 kp/kv/force limits;
- command mode bits required by Unitree low-level control;
- watchdog timestamp / sequence number.

Do not include dexterous hands in this backend. Dex3 or other hand control
must stay on a separate command path unless a policy explicitly outputs hand
actions.

## Safety Staging

1. Passive log only: read LowState at target rate, write no LowCmd.
2. Offline dry-run: feed recorded LowState into `LogReplayG1RobotBackend`.
3. Shadow mode: compute LowCmd from live LowState but do not publish it.
4. Command audit: publish only to a disabled/simulated command sink.
5. Stand target test: fixed/default joint targets with strict delta limits.
6. Short policy test: official-context SONIC body policy, low speed, external
   estop, short horizon, and saved logs.

Hard gates before any motor-enabled command:

- finite state/action/target;
- correct 29D body order;
- target position within joint limits;
- per-frame target delta limit;
- rate watchdog;
- estop/fault abort;
- no dexterous hand command emitted by the body policy route.

# Review

Status: planned, not implemented.

The next code route should add a real-backend skeleton or adapter interface only
after confirming the available Unitree SDK message names and hardware order on
the target robot.
