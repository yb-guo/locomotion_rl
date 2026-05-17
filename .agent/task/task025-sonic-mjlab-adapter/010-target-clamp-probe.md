# 010: Target Clamp Probe

## Route

Test whether the current SONIC/mjlab rollout can run normally if the motor
position targets sent to mjlab are clipped to mjlab soft joint limits.

This is diagnostic only. The formal `MjlabG1RobotBackend` must keep passing raw
SONIC targets through unchanged, so the clamp lives only inside
`mjlab_sonic_alignment_trace.py`.

Expected comparison:

- baseline from `009`: raw SONIC targets can violate mjlab soft limits;
- clamp probe: raw target violation may remain, but sent target violation should
  become zero;
- if root height, pitch, no-done behavior, and foot contacts remain acceptable,
  then target/limit alignment is a plausible next fix;
- if gait worsens, the issue is not only invalid target range.

## Log

- 2026-05-17 Added `--clamp-targets-to-soft-limits` to
  `mjlab_sonic_alignment_trace.py`.

  The trace-only wrapper:

  - reads `robot.data.soft_joint_pos_limits`;
  - clamps `G1MotorCommand.motor_position_targets_mujoco` before converting to
    mjlab `JointPositionAction`;
  - leaves the raw SONIC action unchanged for decoder history;
  - records both raw target limit margins and sent target limit margins;
  - records per-joint `target_clip_delta`, `target_clip_rms`, and
    `target_clip_absmax`.

- 2026-05-17 Local verification passed:

  ```text
  PYTHONPATH=src python -m pytest tests/test_mjlab_sonic_alignment_trace.py -q
  10 passed

  PYTHONPATH=src python -m pytest -p no:cacheprovider
  323 passed, 17 skipped
  ```

- 2026-05-17 The user approved remote sync. H200 direct `git clone` of the PR
  branch failed because GitHub access through `gh-proxy.com` timed out, so the
  updated trace tool was copied into the existing remote adapter source tree.

- 2026-05-17 H200 clamp probe completed, 400 steps:

  ```text
  trace:
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/
    outputs/task025/alignment_trace_target_clamp_400/target_clamp_400.json

  done_steps []
  root_z_mean 0.7627
  root_z_final 0.7465
  abs_pitch_p95 0.1250
  joint_error_rms_mean 0.1486
  root_lin_vel_b_x_mean 0.6916
  planner_root_vel_x_mean 0.6472
  target_clip_absmax_max 1.1098
  target_clip_rms_mean 0.0110
  ```

- 2026-05-17 Clamp probe limit diagnostics:

  ```text
  sent target soft-limit violation fraction:
    all top joints 0.0000

  raw target soft-limit violation fraction:
    left_ankle_roll_joint   0.2075
    left_ankle_pitch_joint  0.0925
    left_knee_joint         0.0775
    right_ankle_pitch_joint 0.0775
    right_knee_joint        0.0250

  top target clip absmax:
    left_ankle_pitch_joint  1.1098 rad
    right_ankle_pitch_joint 0.5096 rad
    left_ankle_roll_joint   0.0936 rad
    left_knee_joint         0.0745 rad
    right_knee_joint        0.0298 rad
  ```

- 2026-05-17 Compared with the previous `009` best unclamped 400-step baseline:

  ```text
  unclamped:
    done_steps []
    abs_pitch_p95 0.1225
    joint_error_rms_mean 0.1527
    root_z_final 0.7468
    left_ankle_pitch_rms 0.4374
    right_ankle_pitch_rms 0.3984

  clamped:
    done_steps []
    abs_pitch_p95 0.1250
    joint_error_rms_mean 0.1486
    root_z_final 0.7465
    left_ankle_pitch_rms 0.3763
    right_ankle_pitch_rms 0.3666
  ```

## Review

The clamped target rollout runs normally under the current 400-step diagnostic
definition: no termination, similar root height, similar pitch, and zero sent
target soft-limit violations.

This is not a full fix yet. Clamping removes infeasible sent targets and reduces
ankle-pitch residual, but it does not materially improve posture:

- `abs_pitch_p95` is effectively unchanged;
- `root_z_final` is effectively unchanged;
- ankle-pitch tracking improves, but ankle pitch remains the largest residual.

This supports target/limit alignment as a real issue, not the only remaining
issue. The next step should compare official SONIC joint ranges against mjlab
G1 ranges and decide whether the right fix is a limit contract patch, an asset
limit patch, or retraining/domain adaptation.

Reproduction command:

```bash
python -m h200_locomotion_lab.tools.mjlab_sonic_alignment_trace \
  --task-id Unitree-G1-Flat \
  --planner /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/gear_sonic_artifacts/planner_sonic.onnx \
  --planner-runner /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/bin/sonic_planner_ort_runner \
  --encoder /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/gear_sonic_artifacts/model_encoder.onnx \
  --decoder /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/gear_sonic_artifacts/model_decoder.onnx \
  --planner-work-dir outputs/task025/alignment_trace_target_clamp_400/planner_work \
  --output-dir outputs/task025/alignment_trace_target_clamp_400 \
  --trace-name target_clamp_400 \
  --steps 400 \
  --replan-interval 10 \
  --device cuda:0 \
  --fixed-base-reset \
  --seed 123 \
  --planner-context-source motion \
  --target-vel 0.5 \
  --sonic-hip-pitch-actuator \
  --clamp-targets-to-soft-limits
```

Decision criteria:

- `done_steps` remains empty;
- `top_joint_target_soft_limit_violation_fraction` is zero or near zero;
- `top_joint_raw_target_soft_limit_violation_fraction` still shows the original
  raw SONIC violations;
- ankle-pitch tracking error improves without a large regression in
  `abs_pitch_p95`, `root_z_final`, or forward velocity.
