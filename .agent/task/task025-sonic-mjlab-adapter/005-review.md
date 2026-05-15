# 005: Review

## Route

Summarize whether SONIC can transfer into `unitree_rl_mjlab`, separating
adapter correctness from policy quality.

## Log

- 2026-05-15 Local implementation tests passed:
  `PYTHONPATH=src python -m pytest -p no:cacheprovider`
  reported `307 passed, 17 skipped`.
- 2026-05-15 H200 `zero` provider smoke rendered successfully, but the robot
  collapsed because SONIC zero raw action maps to the SONIC default pose rather
  than the mjlab trained stance.
- 2026-05-15 H200 synthetic `sequence` provider smoke rendered successfully
  with no terminations over 120 steps. The robot still fell because the replay
  used a repeated fixture row, not an official SONIC action trace.
- 2026-05-15 `ruff` was not run locally because the environment does not have
  `ruff` installed.

## Review

Status: partial pass.

Passed:

- The adapter boundary is modular:
  `ActionProvider -> ScalarG1Runtime -> G1MotorCommand -> G1RobotBackend`.
- mjlab G1 joint-position target names match the SONIC 29DoF MuJoCo command
  order.
- The backend maps targets by joint name rather than assumed index.
- Local tests and H200 smoke runs show finite state/action plumbing and video
  rendering.

Not passed:

- Official SONIC sequence replay is still blocked by missing action/ONNX
  artifacts in the current workspace.
- Online planner/encoder rollout is still blocked for the same reason.
- No stable locomotion claim is made from zero or synthetic fixture actions.
