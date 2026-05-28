# 023: Official Motion Replay Plant Response

## Route

Replay official planner-mode motion through mjlab to isolate plant response:

1. Audit official `planner_motion.csv` and `target_motion.csv` layout and
   choose the correct per-control-step replay source.
2. Add a deterministic mjlab replay trace that commands official motion joint
   positions directly as motor targets and records measured q, base pitch,
   contact, actuator force, and target residuals.
3. Run H200 replay using the official `SLOW_WALK target_vel=0.5`
   `target_motion.csv` generated in `022`.
4. Compare official deploy measured q/base pitch/torque against mjlab replay
   measured q/base pitch/contact/actuator response.

Ranked hypotheses:

1. If mjlab tracks the official target-motion joints with similar q residuals
   and pitch, the remaining mismatch is upstream decoder/planner closed-loop
   distribution rather than mjlab plant response.
2. If mjlab q residuals or base pitch are much worse under the exact official
   target trajectory, the remaining issue is plant/contact/actuator response.
3. If replay immediately terminates or saturates while official deploy does
   not, the mjlab asset/controller/actuator contract is still mismatched even
   without decoder feedback.

## Log

- 2026-05-18 Opened after `022` showed official planner-forward raw servo
  demands are not smaller than mjlab's. The next diagnostic replays the
  official per-step motion target trajectory into mjlab.
- 2026-05-18 Audited official motion CSV layout from source and artifacts:

  ```text
  target_motion.csv:
    no header, one row per deploy control sample
    columns: root xyz, root quat, 29 joint positions in MuJoCo command order
    official writer emits a trailing comma, so CSV readers see 37 cells

  planner_motion.csv:
    no header, same qpos row layout, but it is not one row per control sample
    official writes full planner-motion segments at replans, separated by blank rows
  ```

  Therefore the per-step replay source is `target_motion.csv`; `planner_motion.csv`
  is recorded as planner segment evidence.
- 2026-05-18 Added
  `h200_locomotion_lab.tools.mjlab_official_motion_replay_trace` with two
  replay modes:

  ```text
  target-command:
    command the official motion joint positions directly as mjlab position targets.

  sonic-decoder:
    use the official motion rows as a fixed SONIC encoder motion, run the
    official encoder/decoder, and command mjlab with decoder motor targets.
  ```

  Local regression group passed:
  `PYTHONPATH=src python -m pytest tests/test_mjlab_official_motion_replay_trace.py tests/test_mjlab_sonic_alignment_trace.py tests/test_scalar_action_bridge.py tests/test_sonic_controller.py -q`
  reported `30 passed` with the existing pytest cache permission warning.
- 2026-05-18 H200 `target-command` replay used:

  ```text
  motion:
    /mnt/workspace/users/guoyubo/agent_workspace/official/GR00T-WholeBodyControl/gear_sonic_deploy/outputs/task025/official_planner_slowwalk_0p5_forward_v1/target_motion.csv
  result:
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/official_motion_replay_slowwalk_0p5_400/target_motion_replay_slowwalk_0p5_400.json
  ```

  The result was a severe plant stress test rather than a faithful official
  controller replay:

  ```text
  target-command mjlab:
    done_steps [35, 70, 107, 130, 162, 194, 223, 251, 279, 321, 351, 378]
    abs_pitch_p95 0.8887
    root_z_final 0.2375, root_z_min 0.1029
    joint_error_rms_mean 0.7839
    foot_contact_force_norm_mean [106.24, 123.93]
    foot_contact_force_norm_max [2380.34, 2371.32]
  ```

  This confirms direct reference-motion position replay is not the official
  deploy control contract. Official deploy uses this motion as encoder/current
  motion context, not as the sent PD target.
- 2026-05-18 H200 `sonic-decoder` replay used the same official
  `target_motion.csv` as a fixed SONIC motion input, plus the official
  `model_encoder.onnx` and `model_decoder.onnx`:

  ```text
  result:
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/official_motion_sonic_decoder_replay_slowwalk_0p5_400/sonic_decoder_motion_replay_slowwalk_0p5_400.json
  ```

  Key comparison against the official deploy logs from the same
  `SLOW_WALK target_vel=0.5` run:

  ```text
  official deploy, first 400 rows:
    abs_pitch_p95 0.1548
    joint_error_rms_mean vs official motor targets 0.7858
    top torque abs max: left/right knee 139, hip pitch/roll/yaw 88

  mjlab sonic-decoder replay:
    done_steps [27, 64, 88, 116, 137, 168, 201, 222, 249, 278, 301, 324, 354, 380]
    abs_pitch_p95 1.0294
    root_z_final 0.6301, root_z_min 0.2480
    root_lin_vel_b_x_mean 0.4201
    joint_error_rms_mean 0.7160
    foot_contact_force_norm_mean [115.14, 96.88]
    foot_contact_force_norm_max [1237.95, 1535.93]
    top joint error RMS: waist_yaw 1.3442, left_hip_yaw 1.2974,
      waist_pitch 1.1230
    top actual soft-limit violation fractions:
      left_hip_roll 0.2525, waist_pitch 0.1325, waist_yaw 0.1025,
      left_ankle_pitch 0.0975
    top mjlab actuator force saturation fractions:
      right_wrist_pitch 0.9475, right_wrist_yaw 0.9375,
      left_wrist_yaw 0.8075
  ```

  mjlab joint residual RMS is not worse than official when each is compared to
  its own sent motor targets, but mjlab base pitch and termination behavior are
  much worse.

## Review

Status: passed.

The same-motion replay changes the priority again. Feeding the official
planner-mode motion to mjlab through the SONIC encoder/decoder does not make
the rollout behave like official deploy. The official run stays relatively
upright (`abs_pitch_p95=0.1548`) while mjlab repeatedly terminates and reaches
`abs_pitch_p95=1.0294`, despite comparable or slightly lower aggregate
joint-target residual.

That points away from "the decoder target residual is simply too large" as the
main remaining explanation. The stronger signal is whole-body plant/contact
response: mjlab loses base posture under the same motion-conditioned decoder
route, has repeated soft-limit violations, and reports large contact impulses.
The mjlab actuator-force signal should still be treated cautiously because its
utilization can exceed 1 by large factors, but the contact, pitch, root height,
and done events are enough to prioritize plant/contact/actuator-model
alignment next.

The next diagnostic should focus on lower-body plant response under the
official decoder targets: hip/knee/ankle stiffness/damping/effort, contact
friction/foot geometry, and reset/base state alignment. Target clipping remains
a diagnostic tool, not the production controller contract.
