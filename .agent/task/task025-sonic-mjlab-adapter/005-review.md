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
- 2026-05-15 User explicitly approved restoring official GEAR-SONIC artifacts.
  Local download path:
  `.external_downloads/gear_sonic_artifacts`.
- 2026-05-15 Uploaded official SONIC ONNX artifacts to H200 and verified
  SHA256 for encoder, decoder, and planner.
- 2026-05-15 H200 `online` provider smoke ran with official planner,
  encoder, and decoder artifacts. The 40-step run made 4 planner calls with no
  terminations. The 160-step run made 16 planner calls with no terminations,
  moved forward about 1.27 m, and rendered video.
- 2026-05-15 `ruff` was not run locally because the environment does not have
  `ruff` installed.

## Review

Status: adapter pass, locomotion quality not yet passed.

Passed:

- The adapter boundary is modular:
  `ActionProvider -> ScalarG1Runtime -> G1MotorCommand -> G1RobotBackend`.
- mjlab G1 joint-position target names match the SONIC 29DoF MuJoCo command
  order.
- The backend maps targets by joint name rather than assumed index.
- Local tests and H200 smoke runs show finite state/action plumbing and video
  rendering.
- Official planner, encoder, and decoder artifacts can run end-to-end through
  the mjlab adapter on H200.

Not passed:

- Official SONIC sequence replay is still blocked by missing upstream reference
  motion CSV/directory, if we still want that route.
- No stable locomotion claim is made yet from the online smoke. The 160-step
  run moves forward but root height still drops, so the next diagnosis should
  compare reset/context/command construction against official SONIC.
