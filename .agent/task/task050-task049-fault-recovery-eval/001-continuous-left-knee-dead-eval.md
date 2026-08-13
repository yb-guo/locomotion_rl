# 001 Continuous Left-Knee Dead Eval

## Route

Run the existing Task044 continuous hidden-fault eval on the Task049 checkpoint.
Use hidden `left_knee_joint` dead motor, `vx=1.6 m/s`, onset `2.0 s`, no
recovery, and the unchanged post-fault gate.

## Log

- 2026-08-12 Historical Task044 task id attempt wrote
  `outputs/task050/continuous_left_knee_dead/task049_model9_left_knee_dead_continuous_seed5000101.json`
  but failed setup with a missing registry entry:
  `KeyError('Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenPoseTight1p6')`.
- 2026-08-12 Task048 clean task attempt wrote
  `outputs/task050/continuous_left_knee_dead/task049_model9_task048base_left_knee_dead_continuous_seed5000101.json`
  but failed setup because the clean env had no
  `dynamic_motor_failure` event.
- 2026-08-12 Added and verified
  `Unitree-G1-Gripper-Flat-Task050-TrueTxl-DynamicFault-Eval`, then ran:
  `task044_continuous_fault_eval --task Unitree-G1-Gripper-Flat-Task050-TrueTxl-DynamicFault-Eval --num-envs 256 --steps 360 --seed 5000101 --device cuda:0 --lin-vel-x 1.6 --dynamic-dead-joint left_knee_joint --dynamic-onset-s 2.0 --dynamic-recovery-s 999.0 --startup-excluded-s 0.5 --post-fault-window-s 2.0`.
- Output JSON:
  `outputs/task050/continuous_left_knee_dead/task049_model9_task050_left_knee_dead_continuous_seed5000101.json`.
- Key result: `pipeline_pass=false`, `quality_gate_pass=false`,
  `physical_continuity_pass=false`, `physical_reset_events=512`,
  `physical_fall_events=512`, `memory_debug_active=true`.
- Post-fault window: `sample_count=25600`, `coverage_ratio=1.0`,
  `fall_ratio=1.0`, `lin_vel_error.mean=0.6310434341430664`,
  `yaw_vel_error.mean=0.47820937633514404`,
  `gravity_xy.max=0.9543123841285706`,
  `root_z.min=0.27702611684799194`.
- Failure reasons:
  `physical_continuity_not_preserved`,
  `post_fault_window_quality_not_passed`,
  `post_fault_lin_vel_error_too_high`,
  `post_fault_gravity_xy_too_high`,
  `post_fault_root_z_too_low`.

## Review

Status: complete, failed quality gate.

The Task049 True-TXL checkpoint does not continuously recover from hidden
`left_knee_joint` dead-motor damage in this eval. The runner and actor wiring
were correct (`Task044TrueTxlMemoryK160ContinuousRunner`,
`Task038TrueTxlMemoryModel`, `31` actions), and memory debug was active, but
physical continuity was not preserved.
