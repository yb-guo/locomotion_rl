# 007: Render And Review

## Route

Render the accepted task029 checkpoint and review gait quality. This subtask is
the human-facing sanity check for whether the policy is walking or exploiting
the reward.

Render cases:

- Clean fixed-command walking.
- In-distribution motor-failure sample.
- One or more representative dead-motor grid cases.
- Optional worst passing and worst failing grid cases.

Review focus:

- Excessive high-frequency shaking.
- Foot dragging or skating.
- Upper-body flailing.
- Gripper or arm motion used as a reward exploit.
- Asymmetric gait consistent with the sampled failed motor.
- Falls or near-falls hidden by aggregate metrics.

## Minimal Closed Loop

Feedback loop:

1. Render videos from the exact checkpoint used in 006.
2. Save video, midframe, and render summary JSON.
3. Record command, seed, failure mask, motor scales, and checkpoint path.
4. Add absolute H200 artifact paths and local preview paths if copied.

Pass:

- At least one clean video and one motor-failure video are saved.
- Render summary records the exact checkpoint and fault settings.
- Video review does not show obvious reward hacking in accepted cases.
- Any visible gait defect is documented with the corresponding eval metrics.

Fail:

- Render uses a different checkpoint from eval without explanation.
- Fault settings are not recorded.
- Only still images are produced when video was requested by the task.
- Obvious reward hacking is ignored.

Evidence:

- Planned output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/`.

## Log

- 2026-05-19 Opened because task028 showed render evidence is necessary to
  interpret walking metrics and gripper/upper-body behavior.
- 2026-05-19 Prepared the same-checkpoint render script:
  `.agent/task/task029-motor-only-randomization-and-failure-baseline/artifacts/task029_render_failure_checkpoint.py`.
  It renders clean and in-distribution failure cases by default, with an
  required forced dead-motor case for the final gate. Each case saves an mp4, midframe PNG, and
  per-case JSON; the run also writes an aggregate summary JSON.
- 2026-05-19 Render script JSON records checkpoint path, command, seed, fixed
  command, event/failure settings, video path, midframe path, frame count, FPS,
  byte sizes, done count, and gripper action ranges.
- 2026-05-19 Local checks passed:
  `python .../task029_render_failure_checkpoint.py --help` and AST parse. A
  `py_compile` attempt was not used as evidence because the existing ignored
  `__pycache__` directory denied pyc replacement on Windows.
- 2026-05-19 Copied the script to H200:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/.agent/task/task029-motor-only-randomization-and-failure-baseline/artifacts/task029_render_failure_checkpoint.py`.
  H200 `--help` passed from the MJLab checkout with:
  `PYTHONPATH=/tmp/task029_ipython_stub:/tmp/task029_pydeps:. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true`.
- 2026-05-19 Checkpoint blocker: no task029 `model_*.pt` was found under
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl`
  and no task029 render output exists yet under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/`.
  No render was run because 005 has not provided the same checkpoint used by
  006.
- Planned H200 command once 006 has the same 005 checkpoint:
  ```bash
  cd /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab
  PYTHONPATH=/tmp/task029_ipython_stub:/tmp/task029_pydeps:. \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true \
  /mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
    /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/.agent/task/task029-motor-only-randomization-and-failure-baseline/artifacts/task029_render_failure_checkpoint.py \
    --checkpoint <005-final-model.pt> \
    --output-dir /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/ \
    --prefix task029-render \
    --cases clean failure dead \
    --device cuda:0
  ```
- 2026-05-19 `model_1200.pt` was not available before render start, so 007
  used the same `model_600.pt` checkpoint as 006:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_mlp_baseline_005_train/2026-05-19_20-45-49_005_failure_env8192_iter1200_gpu1_seed29005/model_600.pt`.
  Command:
  ```bash
  cd /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab
  PYTHONPATH=/tmp/task029_ipython_stub:/tmp/task029_pydeps:. \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true \
  /mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
    /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/.agent/task/task029-motor-only-randomization-and-failure-baseline/artifacts/task029_render_failure_checkpoint.py \
    --checkpoint /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_mlp_baseline_005_train/2026-05-19_20-45-49_005_failure_env8192_iter1200_gpu1_seed29005/model_600.pt \
    --output-dir /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/ \
    --prefix task029-render-model600 \
    --cases clean failure dead \
    --device cuda:0
  ```
- 2026-05-19 Render script failure and fix: H200's `mediapy` lacks
  `write_video` / `write_image`, and image fallbacks did not reliably create
  the requested `*-midframe.png` files. The local artifact script was updated
  and synced to H200 with writer fallbacks plus a pure-Python PNG writer and
  post-write stat checks. The final rerun wrote non-empty mp4 and midframe
  files for all required cases.
- 2026-05-19 Render summary JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/task029-render-model600-summary.json`.
  Summary `pass=true`; it records the same checkpoint as 006, command,
  `--cases clean failure dead`, fixed command, per-case fault settings, video
  paths, midframe paths, frame counts, byte sizes, and gripper action ranges.
- 2026-05-19 Render evidence paths:
  - Clean JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/task029-render-model600-clean_forward_0p5.json`.
    Video:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/task029-render-model600-clean_forward_0p5.mp4`.
    Midframe:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/task029-render-model600-clean_forward_0p5-midframe.png`.
  - In-distribution failure JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/task029-render-model600-failure_indistribution_forward_0p5.json`.
    Video:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/task029-render-model600-failure_indistribution_forward_0p5.mp4`.
    Midframe:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/task029-render-model600-failure_indistribution_forward_0p5-midframe.png`.
  - Dead-motor JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/task029-render-model600-dead_motor_left_knee_joint.json`.
    Video:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/task029-render-model600-dead_motor_left_knee_joint.mp4`.
    Midframe:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/task029-render-model600-dead_motor_left_knee_joint-midframe.png`.
- 2026-05-19 Reran render for the final checkpoint
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_mlp_baseline_005_train/2026-05-19_20-45-49_005_failure_env8192_iter1200_gpu1_seed29005/model_1199.pt`,
  matching the final 006 eval checkpoint. Output directory:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/model1199/`.
  Command:
  ```bash
  cd /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab
  PYTHONPATH=/tmp/task029_ipython_stub:/tmp/task029_pydeps:. \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_DISABLED=true \
  /mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python \
    /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/.agent/task/task029-motor-only-randomization-and-failure-baseline/artifacts/task029_render_failure_checkpoint.py \
    --checkpoint /mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_mlp_baseline_005_train/2026-05-19_20-45-49_005_failure_env8192_iter1200_gpu1_seed29005/model_1199.pt \
    --output-dir /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/model1199/ \
    --prefix task029-render-model1199 \
    --cases clean failure dead \
    --device cuda:0
  ```
- 2026-05-19 Final `model_1199.pt` render summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/model1199/task029-render-model1199-summary.json`.
  Summary `pass=true`; all required clean, in-distribution failure, and
  forced dead-motor cases wrote mp4, midframe PNG, and JSON evidence from the
  same checkpoint used by final 006. The dead render forced
  `left_knee_joint` dead, matching the final 006 failing grid joint, and
  recorded `done_count=2` during the 8-second video.
- 2026-05-19 Final `model_1199.pt` render evidence paths:
  - Clean JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/model1199/task029-render-model1199-clean_forward_0p5.json`.
    Video:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/model1199/task029-render-model1199-clean_forward_0p5.mp4`.
    Midframe:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/model1199/task029-render-model1199-clean_forward_0p5-midframe.png`.
  - In-distribution failure JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/model1199/task029-render-model1199-failure_indistribution_forward_0p5.json`.
    Video:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/model1199/task029-render-model1199-failure_indistribution_forward_0p5.mp4`.
    Midframe:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/model1199/task029-render-model1199-failure_indistribution_forward_0p5-midframe.png`.
  - Dead-motor JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/model1199/task029-render-model1199-dead_motor_left_knee_joint.json`.
    Video:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/model1199/task029-render-model1199-dead_motor_left_knee_joint.mp4`.
    Midframe:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/model1199/task029-render-model1199-dead_motor_left_knee_joint-midframe.png`.

## Review

Status: render artifacts complete for the final `model_1199.pt` gate; final
task029 acceptance is not passed because 006 failed.

This is the final task029 acceptance check. The policy can pass numeric eval
and still be rejected here if the video shows unstable or exploitative motion.

The same final checkpoint used by 006 was rendered for clean,
in-distribution failure, and forced dead-motor cases. The artifact-generation
gate is complete for both `model_600.pt` and final `model_1199.pt`: all three
required final cases have mp4, midframe PNG, and JSON evidence, and the final
aggregate render summary is `pass=true`. This does not override the final 006
dead-motor grid failure: task029 remains not passed because
`dead_motor_grid_05_left_knee_joint` failed the zero-fall threshold in the
`model_1199.pt` aggregate eval.
