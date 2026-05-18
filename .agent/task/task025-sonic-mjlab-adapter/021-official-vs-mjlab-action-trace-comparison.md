# 021: Official vs mjlab Action Trace Comparison

## Route

Build a differential feedback loop between the restored official H200 deploy
trace and the mjlab adapter traces:

1. Parse official deploy `action.csv`, `q.csv`, and `dq.csv` from the
   `official_start_control_smoke` run.
2. Convert official raw policy actions to motor targets with the official
   SONIC scale/default formula and compare them against official G1 limits.
3. Parse mjlab adapter raw/effective-history traces from the current best
   clamped baseline.
4. Compare ankle-pitch target range, target-limit violation, action-history
   semantics, and rollout health.

The comparison is a contract/range comparison only. The official smoke uses
the default reference motion `dance_in_da_party_001__A464`; the mjlab adapter
baseline uses the planner-motion, `target_vel=0.5` route.

Ranked hypotheses:

1. If official deploy produces the same ankle-pitch target excursions, the
   mjlab raw target violation is upstream SONIC policy contract rather than an
   adapter-specific target conversion bug.
2. If official deploy stays inside ankle limits while mjlab exceeds them, the
   mjlab closed-loop planner/context/state distribution is driving the decoder
   out of the official deploy distribution.
3. If official raw actions are similar but target conversion differs, the bug
   is in adapter scale/default/joint-order mapping.
4. If mjlab effective-history reduces target excursions but official deploy
   still uses raw-history semantics, production clamping should remain blocked
   unless it mirrors an explicitly chosen new controller contract.

## Log

- 2026-05-18 Opened after `020` produced non-empty official deploy CSV logs.
  The next diagnostic is a direct range/statistics comparison against the
  existing mjlab adapter traces.
- 2026-05-18 Confirmed the official deploy logging contract from source. In
  `CreatePolicyCommand`, official target generation is:

  ```text
  q_target[i] = default_angles[i]
              + action_buffer[isaaclab_to_mujoco[i]] * g1_action_scale[i]
  last_action[i] = action_buffer[i]
  ```

  `action.csv` therefore records raw policy actions in IsaacLab/policy order,
  not already scaled/offset motor targets. The `StateLogger` header comment
  claiming `action.csv` is scaled/offset is stale for this code path.
- 2026-05-18 Ran the H200 comparison script over:

  ```text
  official:
    /mnt/workspace/users/guoyubo/agent_workspace/official/GR00T-WholeBodyControl/gear_sonic_deploy/outputs/task025/official_start_control_smoke/deploy_logs
  mjlab:
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/alignment_trace_effective_history_raw_400/effective_history_raw_400.json
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/alignment_trace_effective_history_effective_400/effective_history_effective_400.json
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/alignment_trace_target_clamp_detail_400/target_clamp_detail_400.json
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/alignment_trace_official_ankle_limit_raw_400/official_ankle_limit_raw_400.json
  result:
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/official_vs_mjlab_compare/summary.json
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/outputs/task025/official_vs_mjlab_compare/key_stats.json
  ```

  The official trace has `1236` CSV rows, `1235` nonzero-control rows, and
  uses reference motion `dance_in_da_party_001__A464`.
- 2026-05-18 Official deploy target-vs-limit stats show that official SONIC
  itself commands targets far outside the G1 XML joint ranges:

  ```text
  official target, left_ankle_pitch:
    range [-1.6760, 2.0923], hard range [-0.87267, 0.5236]
    violation_absmax 1.5687 rad, violation_fraction 0.1854
    measured q range [-0.8263, 0.5381], q_violation_absmax 0.0145
    target_minus_q_rms 0.4425

  official target, right_ankle_pitch:
    range [-1.2472, 1.9789], hard range [-0.87267, 0.5236]
    violation_absmax 1.4553 rad, violation_fraction 0.2389
    measured q range [-0.7111, 0.5898], q_violation_absmax 0.0662
    target_minus_q_rms 0.4405

  official top target-limit violations:
    waist_pitch        violation_absmax 1.9225, fraction 0.6745
    left_ankle_pitch   violation_absmax 1.5687, fraction 0.1854
    right_ankle_pitch  violation_absmax 1.4553, fraction 0.2389
  ```

  This falsifies the hypothesis that "target outside XML joint limit" alone is
  an adapter-only bug.
- 2026-05-18 mjlab clamped raw-history trace is smaller than official in
  absolute target range but has the same qualitative contract:

  ```text
  mjlab raw history, left_ankle_pitch:
    raw target [-0.8083, 1.5636], mjlab soft range [-0.8029, 0.4538]
    raw_violation_absmax 1.1099 rad, raw_violation_fraction 0.0950
    sent target clipped to [-0.8029, 0.4538]
    actual range [-0.6208, 0.0693], joint_error_rms 0.3865
    effective_action_delta_absmax 2.5306

  mjlab raw history, right_ankle_pitch:
    raw target [-0.8354, 0.8149], mjlab soft range [-0.8029, 0.4538]
    raw_violation_absmax 0.3611 rad, raw_violation_fraction 0.0725
    actual range [-0.4841, 0.1054], joint_error_rms 0.3603
  ```

  Rollout health remained stable: no done, `abs_pitch_p95=0.1241`,
  `root_z_final=0.7464`, `joint_error_rms_mean=0.1488`,
  `root_lin_vel_b_x_mean=0.6916`.
- 2026-05-18 mjlab effective-history trace reduced later target excursions
  but did not materially improve rollout quality:

  ```text
  raw history:
    target_clip_absmax_max 1.1099
    effective_action_delta_absmax_max 2.5306
    left_ankle_pitch raw target max 1.5636
    abs_pitch_p95 0.1241, root_z_final 0.7464, joint_error_rms_mean 0.1488

  effective history:
    target_clip_absmax_max 0.6781
    effective_action_delta_absmax_max 1.5461
    left_ankle_pitch raw target max 1.1319
    abs_pitch_p95 0.1216, root_z_final 0.7435, joint_error_rms_mean 0.1492
  ```

  This confirms effective-history feedback is a useful diagnostic knob but not
  a fix for posture/tracking in the current trace.
- 2026-05-18 The official hard ankle-limit overlay remains negative as a fix:

  ```text
  official-hard ankle overlay:
    target_clip_absmax_max 1.0712
    left_ankle_pitch raw target max 1.5948 against high 0.5236
    abs_pitch_p95 0.1247
    root_z_final 0.7438
    joint_error_rms_mean 0.1517
    left_ankle_pitch joint_error_rms 0.4002
  ```

  Widening mjlab ankle pitch from mjlab soft high `0.4538` to official hard
  high `0.5236` only removes about `0.07 rad` of a roughly `1.1 rad` excursion.
- 2026-05-18 Local regression group passed:
  `PYTHONPATH=src python -m pytest tests/test_mjlab_sonic_alignment_trace.py tests/test_scalar_action_bridge.py tests/test_sonic_controller.py -q`
  reported `26 passed` with the existing local pytest cache permission warning.

## Review

Status: passed.

The comparison changes the diagnosis. The adapter is not uniquely producing
illegal ankle-pitch targets: restored official SONIC deploy also sends
ankle-pitch and waist-pitch position targets far outside the official G1 XML
joint ranges, while measured joints stay mostly inside the physical ranges.
The official contract is therefore closer to "policy emits servo demand, lower
plant/physics limits the realized state" than "policy target must be a valid
joint position".

For the mjlab adapter this means:

- Do not promote trace-only target clamping into the formal controller as the
  default. It is not the official deploy contract.
- Do not treat large ankle target-minus-actual residual by itself as proof of a
  mapping bug. Official deploy has the same target-vs-actual shape at ankle
  pitch.
- Keep raw-action history as the fidelity baseline. Official deploy records and
  feeds raw policy action semantics; effective-history is a diagnostic variant.
- The remaining mismatch is posture/style distribution, actuator/plant response,
  and planner/reference-motion comparability, not a solved joint-limit issue.

The next useful comparison should run official deploy in planner mode with a
forward command, or replay the same official reference motion through the mjlab
adapter, so rollout quality can be compared on matched motion input rather than
only target-range contract.
