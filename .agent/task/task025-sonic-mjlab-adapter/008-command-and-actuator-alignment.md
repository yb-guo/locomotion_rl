# 008: Command and Actuator Alignment

## Route

Continue the alignment diagnosis from `007` after `planner_context_source=motion`
became the baseline.

Questions:

1. Does SONIC `target_vel` mean signed metric velocity?
2. Is the high closed-loop speed planner-side or simulator-side?
3. Is the remaining hip/ankle tracking error caused by action amplitude or
   actuator/profile mismatch?

One-variable probes:

- planner-only command sweep;
- enhanced closed-loop trace with planner/root velocity fields;
- `target_vel=0.5` versus `target_vel=-1.0`;
- SONIC action-scale multiplier probe;
- hip-pitch actuator profile override probe.

## Log

- 2026-05-15 Added planner/root velocity metrics to
  `mjlab_sonic_alignment_trace.py`:

  - planner root velocity from current 50 Hz planner motion;
  - actual root body-frame linear velocity from mjlab robot data;
  - mjlab `twist` command for confirmation, even though SONIC does not consume
    it;
  - optional `--sonic-action-scale-mult`;
  - optional `--sonic-hip-pitch-actuator`.

- 2026-05-15 Added `sonic_planner_command_sweep.py`.

  H200 planner-only sweep, initial standing context, `mode=2`, `seed=1234`:

  ```text
  movement_direction=posx:
    target_vel -1.0 -> planner vx 0.525 m/s
    target_vel -0.5 -> planner vx 0.525 m/s
    target_vel  0.5 -> planner vx 0.267 m/s
    target_vel  1.0 -> planner vx 0.525 m/s

  movement_direction=negx:
    target_vel -1.0 -> planner vx -0.520 m/s
    target_vel -0.5 -> planner vx -0.520 m/s
    target_vel  0.5 -> planner vx -0.263 m/s
    target_vel  1.0 -> planner vx -0.520 m/s
  ```

  Interpretation: `movement_direction` controls direction. `target_vel` is not
  a signed metric velocity; in this mode it behaves like a coarse speed/style
  bucket and saturates for `-1.0`, `-0.5`, and `1.0`.

- 2026-05-15 Enhanced closed-loop trace, H200, `seed=123`, fixed-base reset,
  `planner_context_source=motion`, 200 steps:

  ```text
  target_vel=-1.0:
    done_steps []
    planner_root_vel_x_mean 1.066
    root_lin_vel_b_x_mean 1.135
    root_delta_xy_per_s [1.147, -0.050]
    root_z_final 0.755
    abs_pitch_p95 0.251
    joint_error_rms_mean 0.254

  target_vel=0.5:
    done_steps []
    planner_root_vel_x_mean 0.626
    root_lin_vel_b_x_mean 0.647
    root_delta_xy_per_s [0.651, -0.010]
    root_z_final 0.779
    abs_pitch_p95 0.190
    joint_error_rms_mean 0.175
  ```

  The high speed is planner-side. The actual body follows the planner velocity
  closely.

- 2026-05-15 Confirmed `target_vel=0.5`, 400 steps:

  ```text
  done_steps []
  planner_root_vel_x_mean 0.647
  root_lin_vel_b_x_mean 0.686
  root_delta_xy_per_s [0.694, -0.035]
  root_z_mean 0.759
  root_z_final 0.733
  abs_pitch_p95 0.183
  joint_error_rms_mean 0.170
  top errors: ankle_pitch, hip_pitch, knee
  trace outputs/task025/alignment_trace_motionctx_vel0p5_400/motionctx_vel0p5_400.json
  ```

- 2026-05-15 Tested `--sonic-action-scale-mult 0.7`,
  `target_vel=0.5`, 200 steps:

  ```text
  done_steps []
  planner_root_vel_x_mean 0.626
  root_lin_vel_b_x_mean 0.429
  abs_pitch_p95 0.071
  joint_error_rms_mean 0.167
  root_z_mean 0.749
  ```

  This is not a clean fix. It makes the body under-track planner velocity by
  about 31 percent. Lower pitch comes from damping the policy motion, not from
  better planner/body alignment.

- 2026-05-15 Inspected H200 `unitree_rl_mjlab` G1 actuator config:

  ```text
  mjlab hip_pitch actuator:
    target_names_expr ('.*_hip_pitch_joint', '.*_hip_yaw_joint', 'waist_yaw_joint')
    kp 40.179
    kd 2.558
    effort 88.0
    armature 0.01018

  SONIC profile hip_pitch:
    kp 99.098
    kd 6.309
    effort 139.0
    armature 0.02510
  ```

  This explains the earlier hip-pitch action-scale mismatch:
  mjlab treats hip pitch as a 7520_14 class joint, while the SONIC profile
  treats hip pitch as 7520_22.

- 2026-05-15 Tested `--sonic-hip-pitch-actuator`, which moves hip pitch to the
  SONIC 7520_22 actuator profile while keeping the rest of mjlab unchanged.

  H200, `target_vel=0.5`, 200 steps:

  ```text
  done_steps []
  planner_root_vel_x_mean 0.626
  root_lin_vel_b_x_mean 0.675
  root_z_final 0.778
  abs_pitch_p95 0.126
  joint_error_rms_mean 0.158
  ```

  H200, `target_vel=0.5`, 400 steps:

  ```text
  done_steps []
  planner_root_vel_x_mean 0.647
  root_lin_vel_b_x_mean 0.705
  root_delta_xy_per_s [0.717, -0.015]
  root_z_mean 0.764
  root_z_final 0.744
  abs_pitch_p95 0.123
  joint_error_rms_mean 0.153
  top errors: ankle_pitch remains largest; hip_pitch drops below ankle/hip_roll/yaw
  trace outputs/task025/alignment_trace_motionctx_vel0p5_hippitch_400/motionctx_vel0p5_hippitch_400.json
  ```

## Review

Two more causes are now confirmed:

- SONIC `target_vel=-1.0` was not "walk forward at -1 m/s"; in this mode it
  drives a full-speed gait bucket. `target_vel=0.5` is a better first baseline.
- mjlab's G1 hip-pitch actuator profile does not match the SONIC profile. Moving
  hip pitch to the SONIC 7520_22 profile improves pitch and tracking.

Do not use global action-scale reduction as the next fix. It hides pitch by
under-driving the robot and makes actual velocity lag the planner.

Next target:

- make the hip-pitch actuator override a documented optional mjlab config patch,
  not just a trace flag;
- then inspect ankle pitch. After hip-pitch alignment, ankle pitch is the
  dominant residual error, so the next diagnosis should check ankle actuator
  limits, linkage approximation, and torque/force saturation evidence.
