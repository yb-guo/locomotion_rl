# 007: Alignment Diagnosis

## Route

Diagnose why the online SONIC mjlab adapter can walk for 800 steps without
`fell_over`, but stays visibly low/crouched/backward-leaning.

Use the diagnose loop:

1. Keep the 800-step online rollout as the repro.
2. Add sharper alignment metrics before changing behavior.
3. Test one alignment hypothesis at a time.
4. Treat "no termination" as insufficient; posture and tracking quality are the
   pass/fail signal.

Primary metrics:

- root height min/mean/final
- base roll/pitch p95
- actual joint position vs SONIC motor target RMS by joint group
- planner qpos context vs live qpos history
- encoder observation field norms
- raw policy action and mjlab raw action ranges

## Ranked Hypotheses

1. Reset/default-pose mismatch.

   Evidence: mjlab `Unitree-G1-Flat` resets to `HOME_KEYFRAME`, while SONIC
   default angles match the more crouched `KNEES_BENT_KEYFRAME`.

   Prediction: if reset is changed to SONIC default angles and root height, the
   decoder history starts closer to the policy distribution and early posture
   should become less biased.

2. Controller/dynamics mismatch, especially hip pitch.

   Evidence: live mjlab action scale differs from the SONIC profile on
   hip-pitch by about `0.197 rad/action`. The adapter inverts the mjlab action
   term so position targets are still correct, but the underlying mjlab
   actuator gains/effort/armature differ from the SONIC control constants.

   Prediction: if actual-vs-target tracking error is concentrated in
   hip/knee/ankle/waist pitch, gait quality is controller-side, not planner-side.

3. Planner context source or phase mismatch.

   Current implementation replans from live 50 Hz qpos history. The module also
   has a planner-motion context builder that is closer to a pure planner rollout
   feedback path.

   Prediction: if live-context replan shows context discontinuities or drift,
   switching context to previous planner motion with lookahead should reduce
   posture drift.

4. Encoder observation under-filled.

   `observation_config.yaml` marks all encoder fields enabled, but the current
   G1 encoder builder only fills the listed required G1 fields and leaves root-z
   and other enabled fields at zero.

   Prediction: filling root-z and other non-required-but-enabled fields changes
   token/action statistics and may improve height tracking.

5. Evaluation domain randomization mismatch.

   The H200 play config still reports startup randomization terms:
   `foot_friction`, `encoder_bias`, and `base_com`.

   Prediction: disabling startup randomization should make rollout metrics more
   deterministic. If posture improves, the adapter is being evaluated under a
   distribution the official SONIC policy did not assume.

6. Command/mode/height mismatch.

   Current online runs use planner `mode=2`, `target_vel=-1.0`,
   `movement_direction=(1,0,0)`, and planner `height=-1`.

   Prediction: if planner-generated root-z or gait style is already low, then
   changing explicit height/velocity/mode will alter posture even before
   controller changes.

## Log

- 2026-05-15 Repro signal:

  ```text
  800-step online rollout
  done_steps []
  planner_calls 80
  root_delta_xyz [7.964744463562965, 0.1148466169834137, -0.058523595333099365]
  root_end_z 0.7381809949874878
  contact sheet frames 0/200/400/600/799: upright enough to step, low/crouched/backward-leaning
  ```

- 2026-05-15 Read-only H200 alignment probe:

  ```text
  rms_reset_minus_sonic_default: 0.14466941866382432
  rms_action_offset_minus_sonic_default: 0.14466941866382432
  rms_action_scale_minus_sonic_scale: 0.05170452880562437
  ```

  Largest reset/default differences:

  ```text
  left_knee_joint/right_knee_joint: -0.369 rad
  left_elbow_joint/right_elbow_joint: +0.270 rad
  left_hip_pitch_joint/right_hip_pitch_joint: +0.212 rad
  left_ankle_pitch_joint/right_ankle_pitch_joint: +0.163 rad
  ```

  Largest action-scale difference:

  ```text
  left_hip_pitch_joint/right_hip_pitch_joint: +0.1969 rad/action in mjlab vs SONIC profile
  ```

  Live mjlab reset first leg joints:

  ```text
  hip_pitch -0.100, knee 0.300, ankle_pitch -0.200
  ```

  SONIC default first leg joints:

  ```text
  hip_pitch -0.312, knee 0.669, ankle_pitch -0.363
  ```

## Review

Do not start by hand-tuning action scale or speed. The current evidence says
the adapter can run, but multiple contracts are not aligned:

- mjlab reset/action offset uses HOME pose;
- SONIC decoder history subtracts SONIC crouched default;
- mjlab actuator constants are not identical to SONIC control constants;
- encoder input may be missing enabled non-required fields;
- play mode still has startup randomization.

Next implementation should add an alignment trace CLI, then run this ablation
order:

1. baseline trace with target/actual joint errors;
2. disable startup randomization;
3. reset to SONIC default pose;
4. planner context source switch: live qpos vs previous planner motion;
5. fill encoder root-z/non-required enabled fields;
6. explicit planner command sweep: `target_vel`, `height`, `mode`.
