# 005: Closed-Loop Eval And Render

## Route

Reuse the task027 evidence style for `Unitree-G1-Gripper-Flat`: JSON closed-loop
eval first, render second. Do not mark the environment passed from training
reward alone.

## Minimal Closed Loop

Feedback loop:

1. Load a saved checkpoint.
2. Run deterministic fixed-command eval:
   `vx=0.5, vy=0, yaw=0`, 256 envs, 1000 steps.
3. Run randomized holdout eval with the agreed randomization stage enabled.
4. Render one 8-second headless MuJoCo video with EGL.

Pass:

- Deterministic eval passes:
  zero-fall ratio >= 0.95, linear velocity error <= 0.35 m/s, yaw velocity
  error <= 0.35 rad/s, projected-gravity xy <= 0.35.
- Randomized holdout eval has explicit thresholds recorded before running.
- Render video is non-empty, has expected frame count, and has a sampled
  midframe.
- Gripper actions are logged in eval or render summary.

Fail:

- Only training reward is used as evidence.
- Video exists but has no metadata/frame check.
- Eval cannot distinguish body failure from gripper/contact failure.

Evidence:

- Eval JSON and render artifacts under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/eval_render/`.

## Log

- 2026-05-19 Opened during diagnose audit to keep task028 pass criteria aligned
  with task027.

## Review

Status: planned.
