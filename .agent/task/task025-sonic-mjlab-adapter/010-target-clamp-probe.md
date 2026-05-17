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

- 2026-05-17 H200 execution is not yet verified in this subtask. Attempting to
  copy the updated trace tool to `myserver` was blocked by the approval
  reviewer because local source transfer to that remote host was classified as
  external transfer risk. Do not claim the clamp rollout runs until the user
  explicitly approves remote sync or the remote workspace obtains this PR
  through an approved route.

## Review

Local tool behavior is ready for the H200 experiment, but the actual question
"does the clamped target rollout run normally" is still open.

Run this on H200 after remote sync is approved:

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
