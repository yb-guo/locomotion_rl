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

## Review

Status: pending. No validation or video evidence yet.
