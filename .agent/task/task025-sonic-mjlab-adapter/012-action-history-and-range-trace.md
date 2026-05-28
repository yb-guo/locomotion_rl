# 012: Action History and Range Trace

## Route

Follow up on `011` by recording the exact vectors needed to decide whether a
future target clamp can become a real controller feature:

- raw SONIC policy action in policy order;
- effective action after sent-target clamping, mapped back to policy order;
- raw action and effective action in command/joint order;
- raw target and sent target in command/joint order;
- mjlab soft joint limits in command/joint order;
- observed target range versus soft limits.

## Log

- 2026-05-17 Added `ScalarActionBridge.command_targets_to_policy_action`.

  This inverts command-order position targets back into policy-order raw action
  values:

  ```text
  action[policy_index] = (target[command_index] - default[command_index])
                         / action_scale[command_index]
  ```

- 2026-05-17 Extended `mjlab_sonic_alignment_trace.py` rows with:

  ```text
  raw_action
  effective_action
  raw_action_command_order
  effective_action_command_order
  effective_action_delta_command_order
  effective_action_delta_absmax
  raw_target
  target
  soft_joint_pos_limits
  ```

- 2026-05-17 Extended trace summary with:

  ```text
  effective_action_delta_absmax_max
  top_joint_raw_action_absmax
  top_joint_effective_action_absmax
  top_joint_effective_action_delta_absmax
  top_joint_target_range_vs_soft_limits
  ```

- 2026-05-17 Local verification passed:

  ```text
  PYTHONPATH=src python -m pytest \
    tests/test_scalar_action_bridge.py \
    tests/test_mjlab_sonic_alignment_trace.py -q
  21 passed

  PYTHONPATH=src python -m pytest -p no:cacheprovider
  324 passed, 17 skipped
  ```

- 2026-05-17 H200 detailed clamp trace completed:

  ```text
  trace:
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/
    outputs/task025/alignment_trace_target_clamp_detail_400/
    target_clamp_detail_400.json

  done_steps []
  root_z_final 0.7441
  abs_pitch_p95 0.1241
  joint_error_rms_mean 0.1500
  target_clip_absmax_max 1.1116
  effective_action_delta_absmax_max 2.5344
  ```

- 2026-05-17 Observed raw target range versus mjlab soft limits:

  ```text
  left_ankle_pitch_joint:
    soft range       [-0.8029, 0.4538]
    raw target range [-0.8083, 1.5653]
    sent range       [-0.8029, 0.4538]
    raw violation    1.1116 rad

  right_ankle_pitch_joint:
    soft range       [-0.8029, 0.4538]
    raw target range [-0.8354, 0.8631]
    sent range       [-0.8029, 0.4538]
    raw violation    0.4094 rad

  left_ankle_roll_joint:
    soft range       [-0.2356, 0.2356]
    raw target range [-0.2458, 0.3211]
    sent range       [-0.2356, 0.2356]
    raw violation    0.0855 rad

  left_knee_joint:
    soft range       [0.0611, 2.7314]
    raw target range [-0.0115, 1.4653]
    sent range       [0.0611, 1.4653]
    raw violation    0.0726 rad
  ```

- 2026-05-17 Effective action delta after clamping:

  ```text
  left_ankle_pitch_joint   2.5344
  right_ankle_pitch_joint  0.9334
  left_knee_joint          0.2070
  left_ankle_roll_joint    0.1949
  right_knee_joint         0.0992
  ```

- 2026-05-17 Raw action amplitude by joint:

  ```text
  left_ankle_pitch_joint   raw action absmax 4.3968
  right_ankle_pitch_joint  raw action absmax 2.7957
  left_knee_joint          raw action absmax 2.2707
  right_knee_joint         raw action absmax 1.8328
  ```

## Review

The main remaining issue is now sharper:

- mjlab soft ankle pitch high limit is about `0.454 rad`;
- raw SONIC asks left ankle pitch up to `1.565 rad`;
- target clamping changes the effective policy action by up to `2.53` on left
  ankle pitch.

That is too large to treat as a tiny safety clamp. If clamping becomes part of
the controller, the runtime should also update the action history with the
effective clamped action, otherwise the decoder sees an action history that did
not happen.

The observed range also argues that the next question is not "does clamp run?"
but "which contract is wrong?":

- If official SONIC's source asset allows ankle pitch above mjlab's soft limit,
  this is an asset/joint-limit mismatch.
- If official SONIC uses the same limit, the policy/decoder is producing
  out-of-distribution action magnitudes for the current mjlab closed loop.
- If official deploy clips elsewhere, the adapter must mirror that clip and its
  action-history semantics.

Recommended next probe:

1. Inspect official SONIC source asset / deploy controller for joint range or
   action clipping behavior.
2. If a clip exists upstream, mirror both target clipping and effective action
   history.
3. If no clip exists, treat the mjlab G1 soft limits as a sim-to-sim asset
   mismatch and test a trace-only mjlab soft-limit widening patch before any
   production clamp.
