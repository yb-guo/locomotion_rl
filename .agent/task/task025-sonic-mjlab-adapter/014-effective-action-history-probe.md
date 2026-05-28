# 014: Effective Action History Probe

## Route

Answer the question "what if SONIC gets actual environment feedback?" by
separating the feedback paths:

- already-live environment feedback:
  `body_q`, `body_dq`, `base_ang_vel`, and `base_quat` in decoder history come
  from `backend.read_state()`;
- not fully live:
  `planner_context_source=motion` replans from previous planner motion instead
  of live qpos history because that was the better baseline;
- not live by design:
  encoder observation is built from planner motion future windows;
- ambiguous under clamping:
  `last_action` can be raw decoder output or the effective action implied by
  the command actually sent after target clamping.

Implement the narrowest diagnostic switch:

```text
--history-action-source raw|effective
```

Default `raw` preserves the current official-like behavior. `effective` is
available only through the trace-only clamped backend path and writes the
clamped target's inverse policy action into backend state history.

## Log

- 2026-05-17 Added `--history-action-source {raw,effective}` to
  `mjlab_sonic_alignment_trace.py`.

- 2026-05-17 Updated `SoftLimitClampedMjlabG1RobotBackend` to accept the same
  `ScalarActionBridge` used by `ScalarG1Runtime`.

  This matters for `--sonic-action-scale-mult`: effective action inversion must
  use the same scale as command construction.

- 2026-05-17 Effective-history path now does:

  ```text
  clamped_targets = clamp(raw_targets, mjlab_soft_limits)
  history_action = action_bridge.command_targets_to_policy_action(clamped_targets)
  backend._last_command.raw_action_isaaclab = history_action
  ```

  The formal `MjlabG1RobotBackend` is unchanged.

- 2026-05-17 Local verification passed:

  ```text
  PYTHONPATH=src python -m pytest \
    tests/test_mjlab_sonic_alignment_trace.py \
    tests/test_scalar_action_bridge.py \
    tests/test_sonic_controller.py -q

  25 passed
  ```

  Pytest emitted only the existing local `.pytest_cache` permission warning.

- 2026-05-17 Synced the updated trace tool to H200:

  ```text
  /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/src/
  h200_locomotion_lab/tools/mjlab_sonic_alignment_trace.py
  ```

- 2026-05-17 H200 raw-history clamped trace completed:

  ```text
  trace:
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/
    outputs/task025/alignment_trace_effective_history_raw_400/
    effective_history_raw_400.json

  done_steps []
  abs_pitch_p95 0.1241
  root_z_mean 0.7628
  root_z_final 0.7464
  joint_error_rms_mean 0.1488
  root_delta_xy_per_s [0.7027, -0.0192]
  root_lin_vel_b_x_mean 0.6916
  target_clip_absmax_max 1.1099
  effective_action_delta_absmax_max 2.5306
  ```

  Top ankle residuals:

  ```text
  left_ankle_pitch_joint  rms 0.3865
  right_ankle_pitch_joint rms 0.3603
  ```

  Raw target range versus soft limits:

  ```text
  left_ankle_pitch:
    soft_high 0.4538
    raw_target_max 1.5636
    raw_violation_absmax 1.1099
    raw violation fraction 0.0950

  right_ankle_pitch:
    soft_high 0.4538
    raw_target_max 0.8149
    raw_violation_absmax 0.3611
    raw violation fraction 0.0725
  ```

- 2026-05-17 H200 effective-history clamped trace completed:

  ```text
  trace:
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/
    outputs/task025/alignment_trace_effective_history_effective_400/
    effective_history_effective_400.json

  done_steps []
  abs_pitch_p95 0.1216
  root_z_mean 0.7626
  root_z_final 0.7435
  joint_error_rms_mean 0.1492
  root_delta_xy_per_s [0.7020, -0.0120]
  root_lin_vel_b_x_mean 0.6913
  target_clip_absmax_max 0.6781
  effective_action_delta_absmax_max 1.5461
  ```

  Top ankle residuals:

  ```text
  left_ankle_pitch_joint  rms 0.3812
  right_ankle_pitch_joint rms 0.3670
  ```

  Raw target range versus soft limits:

  ```text
  left_ankle_pitch:
    soft_high 0.4538
    raw_target_max 1.1319
    raw_violation_absmax 0.6781
    raw violation fraction 0.0850

  right_ankle_pitch:
    soft_high 0.4538
    raw_target_max 0.6901
    raw_violation_absmax 0.2363
    raw violation fraction 0.0750
  ```

## Review

Changing action history from raw to effective made the decoder produce less
extreme future raw targets under the clamped trace:

- max target clip dropped from `1.1099` to `0.6781`;
- max effective/raw action disagreement dropped from `2.5306` to `1.5461`;
- left ankle-pitch raw target max dropped from `1.5636` to `1.1319`.

But the rollout quality did not materially improve:

- both runs had no `done`;
- `abs_pitch_p95` improved only slightly: `0.1241` to `0.1216`;
- `root_z_final` worsened slightly: `0.7464` to `0.7435`;
- `joint_error_rms_mean` worsened slightly: `0.1488` to `0.1492`;
- left ankle-pitch RMS improved slightly, right ankle-pitch RMS worsened
  slightly;
- forward velocity was unchanged.

Answer to "feed actual environment feedback into SONIC":

The decoder already receives actual environment feedback for joint positions,
joint velocities, base angular velocity, and base orientation. The most direct
missing piece under target clamping is action history: raw action describes
what the decoder asked for, while effective action describes what the backend
actually sent. The effective-history probe reduces subsequent target-range
extremes, which confirms history semantics matter, but it does not fix posture
or overall tracking.

Next decision:

- if clamping becomes production behavior, action history should probably be
  effective, not raw;
- because official SONIC appears to log raw action and does not expose this
  clamp, production clamping should remain blocked until the upstream
  joint-limit/asset contract is resolved;
- the next non-history route should test whether official hard ankle limits
  versus mjlab soft limits are an asset mismatch, or whether the planner/decoder
  is out of distribution in this mjlab loop.

