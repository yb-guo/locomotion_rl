# 002 Unified Speed Baseline Eval

## Route

Evaluate Task030 `model_5349.pt` before any Task031 training.

Baseline matrix:

- Speeds: `0.4`, `0.8`, `1.2`, `1.6`, `2.0 m/s`.
- Cases:
  - clean walking;
  - persistent random motor-failure eval using the Task029 motor-only failure
    distribution;
  - forced persistent dead-motor grid over the 12 leg joints used by Task029 and
    Task030;
  - specified Task030 canonical dynamic switch.
- Checkpoint: Task030 accepted `model_5349.pt`.

Outputs:

- One JSON per speed/case.
- One aggregate JSON with pass count, failed speeds, failed cases, and worst
  metrics.
- Notes in the subtask log identifying the first training bottleneck.

Classification thresholds:

- Clean and persistent cases:
  `zero_fall_ratio >= 0.90`, `lin_vel_error_mean <= 0.8`,
  `yaw_vel_error_mean <= 0.8`, and `gravity_xy_mean <= 0.8`.
- Dynamic switch:
  `zero_fall_ratio >= 0.90`, `recovery_success_ratio >= 0.75`,
  `post_recovery_lin_vel_error_mean <= 0.8`,
  `post_recovery_yaw_vel_error_mean <= 0.8`, and
  `max_gravity_xy_after_onset <= 0.8`.

The baseline is diagnostic. It must not be overwritten by later training runs.

## Log

- 2026-05-21 Planned as the first H200 feedback loop for Task031.
- 2026-05-21 Ran the `model_5349.pt` baseline matrix on H200 with five speed
  bins, clean, random persistent failure, forced 12-joint persistent dead grid,
  and canonical dynamic switch. Aggregate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task031/baseline_model5349_speed_matrix/task031_baseline_model5349_matrix_summary.json`.
- 2026-05-21 Baseline result: clean, random persistent failure, and canonical
  dynamic switch passed at all speed bins. The forced persistent dead grid
  failed at `0.4`, `1.6`, and `2.0 m/s`. Low-speed failures were
  `left_hip_yaw_joint`, `left_hip_roll_joint`, and `right_knee_joint`; high
  speed failures were `right_knee_joint`.

## Review

Status: complete. Baseline JSON evidence exists and shows that Level A training
must focus on forced persistent dead-grid robustness without regressing clean,
random persistent, or specified dynamic-switch behavior.
