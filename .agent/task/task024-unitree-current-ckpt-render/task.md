# Task024: Unitree Current Ckpt Render

## Goal

Render the current Unitree G1 eval checkpoint so the visual behavior can be inspected.

## Route

- Use the remote `unitree_rl_mjlab` checkout instead of the local lab repo, because
  the active checkpoint and mjlab task registration live there.
- Reuse the existing `unitree-rl-mjlab` conda environment and the verified EGL
  setup.
- Run a short headless MuJoCo eval with `render_mode="rgb_array"` and record an
  MP4 through Gymnasium `VideoRecorder`.

## Log

- 2026-05-15 Found current checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-05-15_11-44-54_task024-official-origin-g1-flat-gpu0-12288-u1000/model_999.pt`.
- 2026-05-15 Rendered `Unitree-G1-Flat` for 300 steps on `cuda:0` with
  `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and terminations disabled for a
  continuous inspection clip.
- 2026-05-15 Copied the video and validation frames to the local workspace:
  `outputs/task024/eval_current_ckpt_model_999/model_999_eval-step-0.mp4`.
- 2026-05-15 Added eval-only fixed velocity command controls to
  `.agent/tmp/task024_render_current_ckpt.py`.
- 2026-05-15 Rendered fixed forward command clips with `twist=[1.0, 0.0, 0.0]`.
  The normal action-scale clip is in
  `outputs/task024/eval_current_ckpt_vx1p0_metric/model_999_vx1p0_metric-step-0.mp4`.
  The amplified clip with `action_scale_multiplier=1.3` is in
  `outputs/task024/eval_current_ckpt_vx1p0_scale1p3/model_999_vx1p0_scale1p3-step-0.mp4`.
- 2026-05-15 Rendered a longer normal-policy clip for 1500 steps with
  `twist=[1.0, 0.0, 0.0]` and terminations enabled:
  `outputs/task024/eval_current_ckpt_vx1p0_long1500/model_999_vx1p0_long1500-step-0.mp4`.

## Review

- Verification evidence: decoded video shape is `(300, 480, 640, 3)`, dtype is
  `uint8`, and pixel range is `0..255`.
- The rendered current checkpoint is visible, but the default eval clip looks
  like a low squat/seated posture rather than a convincing walking policy.
- Fixed `vx=1.0` command stayed active from start to end. With normal action
  scale, the root moved by about `[-2.56, +6.51, -0.02]` over 400 steps. With
  `action_scale_multiplier=1.3`, the root moved by about
  `[-5.33, +7.16, -0.02]` over 400 steps.
- Longer 1500-step normal-policy eval kept `done_steps_env0=[]`, with root
  displacement about `[-17.75, +22.14, -0.02]`, reward mean about `0.0589`,
  reward min about `0.00084`, and max action magnitude about `3.74`.
- The viewer camera tracks the robot body, so world-frame displacement is less
  obvious than the leg motion in the video.
