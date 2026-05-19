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
- 2026-05-19 Started two H200 candidate training workers for this eval gate:
  - `Unitree-G1-Gripper-Flat-Control`, GPU0, 8192 envs, seed 52.
  - `Unitree-G1-Gripper-Flat-Combined`, GPU1, 8192 envs, seed 53.
  The Combined run reached the eval gate first, so Control was stopped after
  its latest saved `model_900.pt`; no Control eval was claimed.
- 2026-05-19 Combined training command used tensorboard logging, no W&B,
  `save_interval=100`, and 8192 envs on GPU1. Passing checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task028_eval_render_005/2026-05-19_16-31-34_gpu1_combined_env8192_seed53_full/model_600.pt`.
- 2026-05-19 Deterministic clean eval passed on
  `Unitree-G1-Gripper-Flat-Combined`, 256 envs x 1000 steps (20 s),
  `vx=0.5, vy=0, yaw=0`. JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/eval_render/combined_model600_eval/model_600_forward_0p5_clean_eval.json`.
  Metrics: zero-fall ratio 1.0, mean linear velocity error 0.3099 m/s,
  mean yaw velocity error 0.1254 rad/s, mean projected-gravity xy 0.0468.
- 2026-05-19 Randomized holdout eval also passed on the same checkpoint with
  thresholds recorded before running: zero-fall ratio >= 0.90, linear velocity
  error <= 0.50 m/s, yaw velocity error <= 0.50 rad/s, projected-gravity xy
  <= 0.45. JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/eval_render/combined_model600_eval/model_600_forward_0p5_randomized_holdout_eval.json`.
  Metrics: zero-fall ratio 1.0, mean linear velocity error 0.3189 m/s,
  mean yaw velocity error 0.1263 rad/s, mean projected-gravity xy 0.0467.
- 2026-05-19 Rendered the same checkpoint headlessly with EGL using
  `.agent/task/task028-randomized-wholebody-morphology-env/artifacts/task028_render_checkpoint.py`.
  Render output directory:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/eval_render/combined_model600_render`.
  Video:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/eval_render/combined_model600_render/task028-g1-gripper-combined-model600-vx0p5.mp4`.
  Midframe:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/eval_render/combined_model600_render/task028-g1-gripper-combined-model600-vx0p5-midframe.png`.
  Summary JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/eval_render/combined_model600_render/task028-g1-gripper-combined-model600-vx0p5.json`.
  Evidence: 400 frames, 8.0 s, 50 FPS, 960x720, video size 730226 bytes,
  midframe size 209901 bytes, done count 0. Logged gripper raw action ranges:
  left `[-0.3106, 0.4270]`, right `[-0.1280, 0.4079]`.
- 2026-05-19 Both training workers were stopped after Combined passed. Final
  process checks showed no residual matching training process.

## Review

Status: passed.

The Combined randomized stage has a saved checkpoint that passes both the
deterministic fixed-command eval and a randomized holdout eval with thresholds
recorded before running. The render path produced non-empty video, non-empty
midframe, metadata JSON, and gripper action statistics. This satisfies the
task028 closed-loop evidence gate.

The Control run was stopped because Combined already met the stronger target.
No Control eval pass is claimed.
