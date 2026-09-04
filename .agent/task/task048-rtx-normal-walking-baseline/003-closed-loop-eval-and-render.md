# 003: Closed-Loop Eval and Render

## Route

Load saved checkpoints and run a clean fixed-command rollout at
`vx=0.5, vy=0, yaw=0`. Disable command curriculum, observation corruption,
pushes, and startup dynamics randomization during the deterministic quality
gate. First use a short screen, then require a 256-env x 1000-step full gate.
Render the first full-pass checkpoint for 400 steps at 50 Hz.

Pass thresholds are fixed before evaluation:

- zero-fall ratio >= 0.95;
- mean planar velocity error <= 0.35 m/s;
- mean yaw velocity error <= 0.35 rad/s;
- mean projected-gravity xy norm <= 0.35.

## Log

- 2026-08-19 Added `task048_normal_walk_eval.py` and
  `task048_normal_walk_render.py`, plus focused tests for fixed command,
  deterministic eval configuration, and the predeclared quality gate. Focused
  tests and Ruff checks passed.
- 2026-08-19 `model_100.pt` 16-env x 50-step load/rollout smoke passed the
  execution path with zero falls and finite metrics, but correctly failed the
  quality gate: mean forward velocity was only `0.011 m/s` and planar velocity
  error was `0.492 m/s`. This checkpoint is upright but not walking.
- 2026-08-19 Progression screens showed genuine learning rather than a static
  posture: `model_200` remained near zero speed, `model_300` reached
  `0.063 m/s`, `model_400` reached `0.177 m/s`, and `model_500` reached
  `0.296 m/s`, all on the same 64-env x 250-step screen. `model_400` was the
  first numerical pass, but later checkpoints tracked the command much better.
- 2026-08-19 Detected that the upstream reset randomized initial yaw over
  `[-pi, pi]`. Body-frame tracking metrics were valid, but world-x displacement
  was not directly interpretable. Fixed clean evaluation and rendering to use
  yaw zero, added a focused regression assertion, and reran the final gates.
- 2026-08-19 `model_600.pt` full gate, 256 envs x 1000 steps (20 seconds):
  zero-fall ratio `1.0`, forward velocity `0.434 m/s`, planar velocity error
  `0.155 m/s`, yaw error `0.105 rad/s`, projected-gravity xy mean `0.0175`,
  and mean +x displacement `8.604 m`; pass.
- 2026-08-19 Final `model_649.pt` full gate, 256 envs x 1000 steps: zero-fall
  ratio `1.0` (`0` done events), forward velocity `0.479 m/s`, planar velocity
  error `0.118 m/s`, yaw error `0.137 rad/s`, projected-gravity xy mean
  `0.0224`, and mean +x displacement `9.496 m`; pass. Evidence:
  `outputs/task048/normal_walk/model649_full_256x1000.json`.
- 2026-08-19 Rendered `model_649.pt` for 400 steps at 50 Hz and 960x720. The
  8-second MP4 contains all 400 requested frames, is 934,323 bytes, and reports
  zero done events, mean forward velocity `0.456 m/s`, and +x displacement
  `3.624 m`; render gate passed. Midframe, gait-cycle contact sheet, and video
  were visually inspected; the robot stays upright and alternates support/swing
  legs through the clip. Evidence directory:
  `outputs/task048/normal_walk/render_model649`.
- 2026-08-19 Focused evaluator tests passed (`3 passed`) and Ruff passed after
  the fixed-yaw correction. Final full repository verification passed with
  `716 passed`, critical Ruff checks, `inspect_agent`, and `git diff --check`.

## Review

Status: passed. The best checkpoint passed the predeclared full numerical gate,
and an independent EGL-rendered rollout passed both artifact and visual checks.
