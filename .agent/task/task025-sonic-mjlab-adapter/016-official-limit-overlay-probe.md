# 016: Official Limit Overlay Probe

## Route

Run a minimal mjlab trace-only A/B that replaces mjlab ankle-pitch soft limits
with official SONIC G1 hard limits before target clamping.

This is not a production controller change. It tests whether the remaining
ankle target clipping is mostly caused by mjlab soft limits being narrower than
official SONIC deploy limits.

Planned diagnostic:

```text
left/right ankle pitch official hard range: [-0.87267, 0.5236]
current mjlab soft range: roughly [-0.8029, 0.4538]
current raw target max: about 1.5653
```

Prediction:

- if the issue is just soft-vs-hard limit mismatch, official-limit overlay will
  nearly remove target clipping;
- if raw target remains far above `0.5236`, overlay will reduce clip by only
  about `0.07 rad` and the larger contract mismatch remains.

## Log

- 2026-05-17 Opened as the second diagnostic after official sim2sim contract
  tracing.
- 2026-05-17 Added trace-only
  `--clamp-limit-source {mjlab-soft,official-g1-hard-ankle-pitch}` to
  `mjlab_sonic_alignment_trace.py`.

  Default remains `mjlab-soft`. The official source only changes left/right
  ankle pitch clamp limits:

  ```text
  left_ankle_pitch_joint  [-0.87267, 0.5236]
  right_ankle_pitch_joint [-0.87267, 0.5236]
  ```

  It does not change formal `MjlabG1RobotBackend`.

- 2026-05-17 Local targeted verification passed:

  ```text
  PYTHONPATH=src python -m pytest \
    tests/test_mjlab_sonic_alignment_trace.py \
    tests/test_scalar_action_bridge.py \
    tests/test_sonic_controller.py -q

  26 passed
  ```

- 2026-05-17 H200 raw-history official-ankle-limit overlay completed:

  ```text
  trace:
    /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/
    outputs/task025/alignment_trace_official_ankle_limit_raw_400/
    official_ankle_limit_raw_400.json

  done_steps []
  abs_pitch_p95 0.1247
  root_z_mean 0.7628
  root_z_final 0.7438
  joint_error_rms_mean 0.1517
  root_delta_xy_per_s [0.7056, -0.0208]
  root_lin_vel_b_x_mean 0.6948
  target_clip_absmax_max 1.0712
  effective_action_delta_absmax_max 2.4424
  ```

  Target range versus official ankle-pitch hard limits:

  ```text
  left_ankle_pitch:
    hard range     [-0.87267, 0.5236]
    raw target max 1.5948
    raw violation  1.0712

  right_ankle_pitch:
    hard range     [-0.87267, 0.5236]
    raw target max 0.9063
    raw violation  0.3827
  ```

  Comparison against `014` raw-history mjlab-soft clamp:

  ```text
  mjlab soft:
    target_clip_absmax_max 1.1099
    abs_pitch_p95 0.1241
    root_z_final 0.7464
    joint_error_rms_mean 0.1488
    left_ankle_pitch_rms 0.3865
    right_ankle_pitch_rms 0.3603

  official ankle hard overlay:
    target_clip_absmax_max 1.0712
    abs_pitch_p95 0.1247
    root_z_final 0.7438
    joint_error_rms_mean 0.1517
    left_ankle_pitch_rms 0.4002
    right_ankle_pitch_rms 0.3931
  ```

## Review

Result: hypothesis falsified.

The official hard-limit overlay only reduced the maximum clip by about
`0.039 rad`, while raw target violation remained above `1.07 rad`. It did not
improve posture or tracking; ankle RMS and total joint RMS worsened.

This rules out "mjlab soft ankle-pitch limit is slightly narrower than official
hard limit" as the primary cause. The dominant problem remains that the decoder
is producing targets far beyond both mjlab soft limits and official G1 hard
limits in this closed loop.

Next useful diagnosis is back to `015`: a true official sim2sim target trace
requires a runnable official checkout/environment, or explicit permission to
fetch/build one.
