# 011: Remaining Alignment Diagnosis

## Route

Continue diagnosis after `010` showed that soft-limit clamping lets the current
best rollout run for 400 steps.

Use the existing deterministic trace pair:

- unclamped:
  `outputs/task025/alignment_trace_ankle_probe_best_400/ankle_probe_best_400.json`
- clamped:
  `outputs/task025/alignment_trace_target_clamp_400/target_clamp_400.json`

Questions:

1. After sent targets are valid, what residuals remain?
2. Is the remaining crouch/pitch primarily a controller tracking problem?
3. Are the force-utilization metrics reliable enough to diagnose saturation?
4. Would turning target clamp into a real fix create another observation
   mismatch?

## Log

- 2026-05-17 Offline trace comparison:

  ```text
  unclamped:
    done_steps []
    root_z_mean 0.7637
    root_z_final 0.7468
    abs_pitch_p95 0.1225
    abs_roll_p95 0.1030
    joint_error_rms_mean 0.1527
    root_minus_planner_z_mean 0.0040

  clamped:
    done_steps []
    root_z_mean 0.7627
    root_z_final 0.7465
    abs_pitch_p95 0.1250
    abs_roll_p95 0.1095
    joint_error_rms_mean 0.1486
    root_minus_planner_z_mean 0.0031
  ```

  Interpretation: root height is already close to planner height. The perceived
  low/crouched posture is not fixed by target clamping because the planner
  trajectory itself is low enough for this baseline.

- 2026-05-17 Top residuals after clamping:

  ```text
  left_ankle_pitch_joint   rms 0.3763, mean -0.0636, max 1.0746
  right_ankle_pitch_joint  rms 0.3666, mean -0.1183, max 0.9598
  left_hip_yaw_joint       rms 0.2133, mean +0.0969, max 0.7137
  left_hip_roll_joint      rms 0.2086, mean -0.1039, max 0.4804
  right_hip_roll_joint     rms 0.1835, mean +0.0921, max 0.4971
  right_hip_pitch_joint    rms 0.1804, mean +0.0376, max 0.4501
  right_hip_yaw_joint      rms 0.1774, mean -0.0666, max 0.6393
  right_knee_joint         rms 0.1699, mean +0.0492, max 0.5418
  ```

  Clamp reduces ankle pitch error but does not remove it:

  ```text
  left_ankle_pitch_joint   0.4374 -> 0.3763
  right_ankle_pitch_joint  0.3984 -> 0.3666
  ```

- 2026-05-17 Clip-event split in the clamped trace:

  ```text
  clip_absmax > 0.20 rad:
    n=34
    joint_error_rms_mean 0.2120
    abs_pitch_mean 0.0682
    foot_contact_force_sum_mean 385.5 N

  clip_absmax <= 0.20 rad:
    n=366
    joint_error_rms_mean 0.1427
    abs_pitch_mean 0.0921
    foot_contact_force_sum_mean 320.7 N
  ```

  Interpretation: large clipping events align with higher tracking error and
  larger contact forces, but not with worse pitch. Target infeasibility is a
  tracking/contact symptom; it is not the direct cause of the current body
  pitch metric.

- 2026-05-17 Force-utilization metric check:

  ```text
  actuator_force group mean/max after clamp:
    lower body mean_abs 0.0232, max_abs 0.5054
    waist      mean_abs 0.1751, max_abs 0.8917
    upper body mean_abs 0.4598, max_abs 8.1440

  joint_effort_target:
    all inspected joints 0.0
  ```

  `actuator_force` can exceed the config `effort_limit` for wrists and
  shoulders, while `joint_effort_target` is zero in this position-actuator path.
  Therefore the current force-utilization summary is useful as a "large actuator
  force" heuristic, but not yet valid proof of physical torque saturation.

- 2026-05-17 Correlation checks after clamp:

  ```text
  corr(abs_pitch, joint_error_rms)     -0.284
  corr(abs_pitch, contact_force_sum)   -0.135
  corr(abs_pitch, lower_force_max)     +0.284
  corr(abs_pitch, upper_force_max)     -0.155
  corr(root_z, planner_root_z)         +0.777
  ```

  Interpretation: remaining pitch is weakly related to lower-body effort, but
  not strongly explained by target clipping, contact force, or upper-body force.

## Review

Additional issues found:

1. **Raw policy/planner target range remains out of contract.**

   Clamp makes sent targets valid, but raw SONIC still asks for invalid ankle
   and knee targets. This points upstream: planner/encoder/decoder distribution,
   SONIC joint limits, or mjlab asset limits are still misaligned.

2. **A production target clamp would need an action-history fix.**

   The trace-only clamp deliberately preserves `raw_action_isaaclab` in
   `last_action` so the diagnostic can expose raw policy behavior. If clamping
   becomes a real controller feature, decoder history should probably contain
   the effective clamped action converted back into policy order, otherwise the
   policy observes actions that were never executed.

3. **Force-utilization metrics need validation before diagnosing saturation.**

   The current metric divides `robot.data.actuator_force` by config
   `effort_limit`, but `joint_effort_target` is zero and wrist utilization can
   exceed 8x. Before claiming saturation, inspect the actual MuJoCo/mjlab
   actuator force range or add a runtime field that exposes the clipped actuator
   command.

4. **Crouch/pitch is not primarily solved by target validity.**

   Root z tracks planner z closely, and clamping barely changes `abs_pitch_p95`
   or `root_z_final`. Next posture work should focus on planner trajectory
   style/context, encoder field completeness, or sim-to-sim dynamics, not only
   soft-limit handling.

Recommended next probes:

- record full raw action and effective clamped action vectors in trace rows;
- compare official SONIC joint ranges with mjlab `soft_joint_pos_limits`;
- validate actuator force limits against the resolved MuJoCo/mjlab model, not
  only config objects;
- test a production-style effective-action history for clamped targets;
- then test encoder root-z / lower-body optional fields only after the action
  and limit contract is clear.
