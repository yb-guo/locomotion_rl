# 022: Matched-Input Official Planner Comparison

## Route

Move from range-contract comparison to matched-input comparison:

1. Inspect the official deploy input interface and keyboard controls for
   planner mode, start, play, and forward command semantics.
2. Run official Python MuJoCo sim plus official `g1_deploy_onnx_ref` in planner
   mode with a forward command.
3. Record non-empty official deploy CSV logs and confirm the run is actually
   planner-driven rather than reference-motion playback.
4. Compare official planner-forward `action/q/dq/token_state` stats against the
   current mjlab planner-motion baseline.

Ranked hypotheses:

1. If official planner-forward target/action ranges and measured q response
   resemble mjlab, the remaining mismatch is not a controller I/O bug; focus on
   plant/contact/actuator response and matched rollout quality.
2. If official planner-forward raw actions are much smaller or phase-shifted
   than mjlab, the mjlab planner/context path is still out of distribution.
3. If official planner mode cannot be reliably driven by non-interactive
   keyboard stdin, use a reference-motion replay comparison as the next matched
   input route.

## Log

- 2026-05-18 Opened after `021` showed official deploy itself commands
  out-of-range ankle targets. The next step is to compare with a closer input
  condition than the default official reference motion.
- 2026-05-18 Inspected official keyboard planner input semantics. Enter
  toggles planner mode, `]` starts control, `1` selects `SLOW_WALK`, `2`
  selects `WALK`, `0` increments speed, and `w` applies forward movement.
  Since the default planner movement mode is `IDLE`, sending only `w` is not a
  forward locomotion command. The closest matched input to the mjlab
  `target_vel=0.5` baseline is `Enter`, `]`, `1`, `0`, `0`, `0`, `w`, which
  logs as `SLOW_WALK, target_vel: 0.5, movement: [1, 0, 0]`.
- 2026-05-18 The first official planner-forward harness failed before deploy
  startup: remote `scripts/setup_env.sh` had CRLF line endings in this shell,
  `just` was not on PATH, and the passed `\n]2w` sequence became literal
  `n]2w`. Reworked the harness to invoke
  `target/release/g1_deploy_onnx_ref` directly, set the TensorRT
  `root/lib` path explicitly, use the bundled Unitree DDS library path, and
  construct the key sequence inside the remote script.
- 2026-05-18 Ran official Python MuJoCo sim plus restored official C++ deploy
  in two planner-forward modes:

  ```text
  official_planner_walk_mode2_forward_v3:
    key_sequence $'\n]2w'
    deploy_status 124, sim_status 124
    planner_control_fraction 1.0
    action/q/dq/token_state rows 1187
    deploy log: Replanning with mode: WALK, target_vel: -1,
      movement: [1, 0, 0]

  official_planner_slowwalk_0p5_forward_v1:
    key_sequence $'\n]1000w'
    deploy_status 124, sim_status 124
    planner_control_fraction 1.0
    action/q/dq/token_state rows 1240
    deploy log: Replanning with mode: SLOW_WALK, target_vel: 0.5,
      movement: [1, 0, 0]
  ```

  `124` is the expected timeout status from the bounded run, not a startup
  failure. Both runs wrote non-empty CSV logs and switched `motion_name` to
  `planner_motion`.
- 2026-05-18 Compared official reference, official planner-forward, and mjlab
  planner-motion traces with:

  ```text
  result:
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/official_matched_input_compare/summary.json
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/official_matched_input_compare/key_stats.json
  ```

  The official deploy logs expose base quaternion, joint state, action, and
  token state but not root position; root height and xy velocity comparisons
  are therefore available only on mjlab traces.
- 2026-05-18 Key matched-input stats:

  ```text
  official planner SLOW_WALK target_vel=0.5:
    control_rows 1239, planner_control_fraction 1.0
    base_pitch_abs_p95 0.1446
    left_ankle target range [-2.0521, 2.2490]
    left_ankle target_violation_absmax 1.7254
    left_ankle q range [-0.8530, 0.6722]
    left_ankle target_minus_q_rms 0.4215
    right_ankle target range [-2.2383, 1.8628]
    right_ankle target_violation_absmax 1.3656
    right_ankle q range [-0.9027, 0.6038]
    right_ankle target_minus_q_rms 0.3863

  official planner WALK mode 2:
    control_rows 1186, planner_control_fraction 1.0
    base_pitch_abs_p95 0.1374
    left/right ankle target_violation_absmax 2.0161 / 1.5134

  mjlab best unclamped motion-context target_vel=0.5:
    steps 400, done_steps []
    abs_pitch_p95 0.1229
    root_z_final 0.7443, root_z_mean 0.7636
    root_delta_xy_per_s [0.7167, -0.0150]
    root_lin_vel_b_x_mean 0.7051
    joint_error_rms_mean 0.1528
    left/right ankle joint_error_rms 0.4369 / 0.4004

  mjlab clamped raw-history diagnostic:
    steps 400, done_steps []
    abs_pitch_p95 0.1241
    root_z_final 0.7464
    left ankle raw target range [-0.8083, 1.5636]
    left ankle raw violation fraction 0.0950
    left/right ankle joint_error_rms 0.3865 / 0.3603
  ```

  The official planner-forward raw target excursions are not smaller than the
  mjlab trace. They are generally larger, while the measured official ankle
  `q` range stays much closer to the physical joint range.

## Review

Status: passed.

Matched-input planner comparison falsifies the hypothesis that mjlab's planner
context alone is making decoder outputs abnormally large. Under an official
planner-forward command, official deploy also emits large raw servo demands:
ankle-pitch target-limit violations are `1.37-2.02 rad`, larger than the
current mjlab raw-history diagnostic's left ankle `1.11 rad` soft-limit clip
need. The official plant absorbs much of that demand in measured `q`, producing
ankle target-minus-q RMS around `0.39-0.42 rad`, comparable to mjlab ankle
tracking RMS.

The next useful route is not to make target clipping the production controller
contract. The remaining question is why the official plant and mjlab plant
absorb the same style of SONIC servo demand into different whole-body posture
and tracking. The next diagnostic should replay or sample the official
`planner_motion.csv` / `target_motion.csv` through mjlab, or otherwise compare
plant/contact/actuator response under the same target trajectory.
