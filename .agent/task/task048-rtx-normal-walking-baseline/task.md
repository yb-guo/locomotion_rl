# Task 048: RTX Normal-Walking Baseline

## Route

Train a normal forward-walking policy from scratch on the new single RTX 5060
Ti using the already verified official `Unitree-G1-Flat` MJLab environment and
MLP/PPO runner. The user explicitly reduced the target to a working walking
policy, so Task044 hidden-fault and retry-context migration is out of scope.

Slices:

1. `001-rtx-throughput-and-budget.md`
   - Probe practical environment counts on the 16 GiB GPU.
   - Choose a stable training budget from measured throughput and memory.
2. `002-from-scratch-training.md`
   - Train from scratch with local TensorBoard logging and periodic checkpoints.
   - Stop after a saved checkpoint passes the walking gate.
3. `003-closed-loop-eval-and-render.md`
   - Evaluate a fixed `vx=0.5, vy=0, yaw=0` command for 20 seconds.
   - Require stability and tracking metrics, then render an 8-second video.

Acceptance gate:

- zero-fall ratio >= 0.95;
- mean planar velocity error <= 0.35 m/s;
- mean yaw velocity error <= 0.35 rad/s;
- mean projected-gravity xy norm <= 0.35;
- non-empty saved checkpoint and headless render video.

## Log

- 2026-08-19 Opened after the user clarified that a normally walking trained
  policy is sufficient. The known-good official G1 flat task is selected to
  minimize time-to-feedback.
- 2026-08-19 Measured the new RTX 5060 Ti, selected 4096 parallel environments,
  and completed a 650-iteration from-scratch PPO run in `00:20:35`.
- 2026-08-19 Selected `model_649.pt` after checkpoint progression tests. In the
  final 256-env x 1000-step clean gate it sustained the requested `0.5 m/s`
  command for 20 seconds with zero falls, mean forward velocity `0.479 m/s`,
  mean planar error `0.118 m/s`, and mean +x displacement `9.496 m`.
- 2026-08-19 Headless EGL rendering produced an 8-second, 400-frame MP4. The
  single rendered robot had zero resets, mean forward velocity `0.456 m/s`,
  and +x displacement `3.624 m`. Multi-frame visual inspection confirmed an
  upright alternating gait throughout the clip.
- 2026-08-19 Packaged the selected PyTorch checkpoint, ONNX policy, agent and
  environment configs, hashes, and metrics under
  `outputs/task048/normal_walk/deliverable`. The copied checkpoint and ONNX
  matched the training outputs byte-for-byte; ONNX checker passed (IR 8,
  opset 18).
- 2026-08-19 Final verification: full repository pytest `716 passed` (35
  upstream TorchScript deprecation warnings), critical Ruff selection passed,
  `inspect_agent` passed, and `git diff --check` passed.

## Review

Status: passed. Training completed without OOM/NaN, the predeclared numerical
gate passed at full scale, and the independently rendered rollout passed.
