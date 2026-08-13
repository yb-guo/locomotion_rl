# 002 Retry Left-Knee Dead Eval

## Route

Run the existing Task037 multi-trial eval on the Task049 checkpoint. Use hidden
`left_knee_joint` dead motor, `vx=1.6 m/s`, onset `0.5 s`, no recovery, three
2.0 s trials, and final-trial promotion metrics.

## Log

- 2026-08-12 After subtask 001 established the Task050 dynamic-fault eval task,
  ran:
  `task037_multitrial_eval_checkpoint --task Unitree-G1-Gripper-Flat-Task050-TrueTxl-DynamicFault-Eval --num-envs 256 --steps 300 --seed 5000201 --device cuda:0 --trial-length-s 2.0 --lin-vel-x 1.6 --dynamic-dead-joint left_knee_joint --dynamic-onset-s 0.5 --dynamic-recovery-s 999.0 --final-window-s 1.0`.
- Output JSON:
  `outputs/task050/retry_left_knee_dead/task049_model9_task050_left_knee_dead_retry_seed5000201.json`.
- Wiring: `runner_cls=Task038TrueTxlMemoryK160Runner`,
  `actor_model_class=Task038TrueTxlMemoryModel`, `action_dim=31`,
  `total_action_dim=31`, `eval_mode=dynamic_single_onset`, and TXL debug
  recorded `total_actor_forward_samples=76800`.
- Trial metrics:
  - `trial_0`: `fall_ratio=1.0`, `fall_count=512`,
    `lin_vel_error.mean=1.2791528701782227`,
    `yaw_vel_error.mean=1.274516224861145`,
    `gravity_xy.max=0.9468333721160889`,
    `root_z.min=0.18728843331336975`.
  - `trial_1`: `fall_ratio=1.0`, `fall_count=256`,
    `lin_vel_error.mean=1.557944655418396`,
    `yaw_vel_error.mean=0.764865517616272`,
    `gravity_xy.max=0.9445145726203918`,
    `root_z.min=0.2236713469028473`.
  - `final_trial`: `fall_ratio=1.0`, `fall_count=256`,
    `lin_vel_error.mean=1.6589775085449219`,
    `yaw_vel_error.mean=1.059837818145752`,
    `gravity_xy.max=0.9453970789909363`,
    `root_z.min=0.23644568026065826`.
- Final window: `fall_ratio=1.0`, `completion_ratio=0.1328125`,
  `lin_vel_error.mean=1.6596848964691162`,
  `root_z.min=0.23644568026065826`.
- Result: `final_trial_pass=false`, `pass=false`.

## Review

Status: complete, failed promotion gate.

The retry route does not show context-enabled recovery after falls. The final
trial is worse than the gate on fall ratio, linear velocity error, yaw velocity
error, gravity xy, and root height. This is negative evidence for the Task049
checkpoint under hidden left-knee dead-motor damage, not a statement about all
damaged joints or a retrained fault-specialized policy.
