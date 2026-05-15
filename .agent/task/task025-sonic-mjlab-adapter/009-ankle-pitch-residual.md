# 009: Ankle Pitch Residual

## Route

Continue from the best `008` baseline:

- `planner_context_source=motion`
- `target_vel=0.5`
- trace-only SONIC hip-pitch actuator profile
- `seed=123`
- 400 mjlab steps

Question: after fixing planner context, command bucket, and hip-pitch actuator
profile, why does ankle pitch remain the dominant tracking residual?

Instrumentation:

- actual joint position versus SONIC motor target, by joint;
- mjlab `actuator_force` and effort-limit utilization;
- actual and target distance to mjlab soft joint limits;
- feet contact force norm from the mjlab contact sensor.

## Log

- 2026-05-15 Added ankle-focused trace fields to
  `mjlab_sonic_alignment_trace.py`:

  - `actuator_force`;
  - `joint_effort_target`;
  - `effort_limits`;
  - `actuator_force_utilization`;
  - `actual_soft_limit_margin`;
  - `target_soft_limit_margin`;
  - `foot_contact_force_norm`.

- 2026-05-15 H200 runtime inspection confirmed mjlab exposes the required data:

  ```text
  robot.data.actuator_force Tensor [1, 29]
  robot.data.joint_effort_target Tensor [1, 29]
  robot.data.soft_joint_pos_limits Tensor [1, 29, 2]
  feet_ground_contact.data.force Tensor [1, 2, 3]
  feet_ground_contact.data.found Tensor [1, 2]
  ```

- 2026-05-15 H200 ankle probe, 400 steps:

  ```text
  trace:
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/
    outputs/task025/alignment_trace_ankle_probe_best_400/ankle_probe_best_400.json

  done_steps []
  root_z_mean 0.7637
  root_z_final 0.7468
  abs_pitch_p95 0.1225
  joint_error_rms_mean 0.1527
  root_lin_vel_b_x_mean 0.7071
  planner_root_vel_x_mean 0.6472
  foot_contact_force_norm_mean [166.4, 163.5] N
  foot_contact_force_norm_max [681.2, 605.5] N
  ```

- 2026-05-15 Remaining top tracking errors:

  ```text
  left_ankle_pitch_joint   rms 0.4374
  right_ankle_pitch_joint  rms 0.3984
  left_hip_roll_joint      rms 0.2133
  left_hip_yaw_joint       rms 0.2047
  right_hip_roll_joint     rms 0.1820
  right_hip_pitch_joint    rms 0.1771
  left_hip_pitch_joint     rms 0.1640
  left_knee_joint          rms 0.1597
  ```

- 2026-05-15 The actuator-force evidence does not support ankle torque
  saturation as the first explanation. Force utilization and saturation are
  dominated by wrists, arms, and waist. Ankle pitch does not appear in the top
  force-utilization or saturation rows.

- 2026-05-15 The actual ankle positions stay inside mjlab soft joint limits:

  ```text
  target soft-limit min margin:
    left_ankle_pitch_joint  -0.9014
    right_ankle_pitch_joint -0.3539
    right_ankle_roll_joint  -0.1003
    left_ankle_roll_joint   -0.0663
    left_knee_joint         -0.0545

  actual soft-limit min margin:
    left_ankle_pitch_joint   0.2540
    right_ankle_pitch_joint  0.3229
  ```

- 2026-05-15 Recomputed the violation split from the same JSON trace:

  ```text
  target_soft_limit_margin violation fraction:
    left_ankle_roll_joint   0.1925
    left_ankle_pitch_joint  0.0925
    right_ankle_pitch_joint 0.0700
    left_knee_joint         0.0675
    right_ankle_roll_joint  0.0275

  actual_soft_limit_margin violation fraction:
    all joints 0.0000
  ```

- 2026-05-15 Error split by target soft-limit violation:

  ```text
  left_ankle_pitch_joint:
    target violation: n=37, rms_abs_error=1.1506, max_abs_error=1.8541
    target valid:     n=363, rms_abs_error=0.2754, max_abs_error=0.9497

  right_ankle_pitch_joint:
    target violation: n=28, rms_abs_error=0.9507, max_abs_error=1.1788
    target valid:     n=372, rms_abs_error=0.3204, max_abs_error=0.8603
  ```

## Review

The ankle-pitch residual is not primarily an actuator saturation signal in this
trace. The stronger explanation is target infeasibility:

- the SONIC/mjlab pipeline sometimes commands ankle targets outside mjlab soft
  joint limits;
- the actual joint remains inside the limits;
- ankle-pitch tracking error spikes strongly when the target is outside the
  soft limits.

This means the next fix should align the target/limit contract before changing
encoder fields or globally scaling actions. Candidate next probes:

- clamp SONIC motor targets to mjlab soft limits only for trace diagnosis and
  check whether posture and no-done behavior remain stable;
- compare official SONIC ankle joint limits against mjlab G1 ankle limits;
- inspect whether the ankle linkage/body/contact model differs enough that the
  planner expects a range mjlab cannot physically execute.
