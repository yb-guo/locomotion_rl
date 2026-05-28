# 005 Full Validation And Video

## Route

Run representative validation for the best curriculum checkpoint and compare it
against Task034 `model_5350.pt`.

Required representative matrix:

- speeds `0.4`, `1.2`, `2.0 m/s`;
- dynamic switch;
- full 12-joint forced dead-grid;
- clean/no-failure sanity if a clean eval helper is available.

Required representative videos:

- `0.4 m/s` clean;
- `1.2 m/s` dynamic switch;
- `2.0 m/s` right-knee forced dead;
- `2.0 m/s` full dead-grid representative case.

Video review criteria:

- no collapse hidden by reset averaging;
- no standing still while satisfying velocity thresholds;
- no obvious foot dragging or high-frequency jitter cheat;
- upper body and gripper posture remain reasonable.

## Log

- 2026-05-28 Planned.
- 2026-05-28 Added full-validation artifact:
  `task035_validate_checkpoint_matrix.sh`.
- 2026-05-28 Ran representative full validation for Task035 `model_5369.pt`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task035/full_validation_model_5369/task035_full_validation_summary.json`.
  Result: `pass=false`. `1.2 m/s` and `2.0 m/s` both pass dynamic switch and
  full 12-joint dead-grid. `0.4 m/s` passes dynamic switch but fails full
  dead-grid with `7/12` pass count. Failed joints:
  `left_hip_pitch_joint`, `left_hip_yaw_joint`, `left_hip_roll_joint`,
  `right_hip_roll_joint`, `right_knee_joint`.
- 2026-05-28 Ran `0.4 m/s` comparison for baseline `model_5350.pt`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task035/full_validation_model_5350_vx0p4/task035_full_validation_summary.json`.
  Result: `pass=false`, dead-grid `9/12`. Failed joints:
  `left_hip_yaw_joint`, `left_hip_roll_joint`, `right_knee_joint`.

## Review

Status: failed full validation. Video review is not required for promotion
because no checkpoint is promotable under the numeric gate.
