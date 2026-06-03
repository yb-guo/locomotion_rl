# 001 Contract And Registration

## Route

Register one comparable H200 task family for each policy consumer:

- AdaptK4;
- GRU K4;
- Token K4.

Each family must have:

- a train task on the existing unified dynamic/failure environment;
- a dynamic-switch eval task;
- a focused forced-deadgrid eval task.

No task may expose explicit fault labels, masks, or scales to the actor.

## Log

- 2026-05-28 Added `task036_register_policy_quality_stages.py`.
- 2026-05-28 Registered H200 task ids under the MJLab adapter checkout:
  `Unitree-G1-Gripper-Flat-Task036-{AdaptK4,GruK4,TokenK4}-Fast2p0`,
  `Unitree-G1-Gripper-Flat-Task036-{AdaptK4,GruK4,TokenK4}-FocusedDeadGrid-Fast2p0`,
  and
  `Unitree-G1-Gripper-Flat-Task036-{AdaptK4,GruK4,TokenK4}-DynamicMotorFailure-Fast1p6`.

## Review

Status: registered. Training/eval quality is not proven by registration.
