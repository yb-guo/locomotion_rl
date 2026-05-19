# Task 027: Unitree MJLab G1 From-Scratch Velocity Baseline

## Route

Run the upstream Unitree MJLab `Unitree-G1-Flat` velocity task as the
known-good from-scratch walking baseline before porting anything into this
repo.

1. Verify the H200 environment can import and launch Unitree MJLab training.
2. Run a small `Unitree-G1-Flat` smoke test.
3. Probe throughput across practical `num_envs` / GPU settings.
4. Start the best-performing baseline run and record its log directory.
5. Do not claim walking quality until a checkpoint is rendered or evaluated.

## Log

- 2026-05-19 Opened after the user asked to start the Unitree MJLab
  from-scratch G1 walking baseline and maximize H200 performance.
- 2026-05-19 Verified upstream `Unitree-G1-Flat` launches on H200 from
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab`
  with `PYTHONPATH=.` and
  `/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python`.
- 2026-05-19 Found the default W&B logger is not usable in the headless
  training session without an API key. The working launch uses
  `--agent.logger=tensorboard --agent.upload-model=False`.
- 2026-05-19 Smoke test passed with 64 envs for 2 PPO iterations. Log dir:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity_task027_smoke/2026-05-19_11-04-47_env64_iter2_tb`.
  Observed steps/s: 872 then 1511.
- 2026-05-19 Single-GPU performance probes on physical GPU0:
  `1024 envs` reached about 33.5k steps/s, `2048 envs` about 59.4k
  steps/s, `4096 envs` about 95.9k steps/s, `8192 envs` about 122.9k
  steps/s, and `16384 envs` about 120.8k steps/s. Best practical point is
  8192 envs per GPU because 16384 does not improve throughput.
- 2026-05-19 Tried the upstream multi-GPU launcher path with
  `--gpu-ids "[0,1]"`, but `torchrunx` failed with a distributed store
  timeout: only 1/2 clients joined. This is a launcher/DDP issue, not a
  simulation-memory limit.
- 2026-05-19 Verified the faster H200 strategy is two independent single-GPU
  jobs instead of one DDP job. Two 4096-env jobs reached about 180.9k total
  steps/s. Two 8192-env jobs reached about 241.8k total steps/s in the
  probe.
- 2026-05-19 Started the formal long baseline as two detached independent
  runs:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/.agent/tmp/task027_start_unitree_mjlab_full.sh`.
  The runs use 8192 envs per GPU, seeds 42 and 43, tensorboard logging,
  `max_iterations=10001`, and `save_interval=100`.
- 2026-05-19 Full-run process evidence:
  `gpu0_env8192_seed42_full pid=506403`,
  `gpu1_env8192_seed43_full pid=506406`. Stdout files:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task027/unitree_mjlab_full/gpu0_env8192_seed42_full.stdout`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task027/unitree_mjlab_full/gpu1_env8192_seed43_full.stdout`.
- 2026-05-19 Full-run status check after launch showed both processes still
  running at learning iteration 62/10001. Recent throughput was about
  116.7k steps/s on GPU0 and 116.3k steps/s on GPU1. Training log dirs:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity_task027_from_scratch/2026-05-19_11-21-28_gpu0_env8192_seed42_full`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity_task027_from_scratch/2026-05-19_11-21-28_gpu1_env8192_seed43_full`.
- 2026-05-19 Later status checks showed the full runs still alive past the
  first save point at learning iteration 110/10001. Both runs saved
  `model_100.pt`:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity_task027_from_scratch/2026-05-19_11-21-28_gpu0_env8192_seed42_full/model_100.pt`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity_task027_from_scratch/2026-05-19_11-21-28_gpu1_env8192_seed43_full/model_100.pt`.
  Throughput fluctuated during the early learning/reset regime, with recent
  values ranging from about 52k to 112k steps/s per GPU; `nvidia-smi` sampled
  87-97% utilization during the heavier segment. The checkpoint confirms the
  train-and-save path is working.
- 2026-05-19 Added a task027 headless closed-loop eval CLI:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/.agent/tmp/task027_unitree_mjlab_closed_loop_eval.py`.
  The CLI loads `model_*.pt`, runs a vectorized closed-loop rollout, and
  writes JSON metrics for fall count, commanded velocity tracking, yaw
  tracking, root height, projected gravity, reward, and pass/fail.
- 2026-05-19 Verified eval `--help` on H200. CPU eval was too slow and left a
  residual test process, which was stopped. GPU eval smoke on `model_100.pt`
  succeeded for 16 envs x 50 steps and wrote
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task027/unitree_mjlab_eval/smoke_model100_gpu0.json`.
  It did not pass because the 0.5 m/s forward velocity error was about
  0.507 m/s.
- 2026-05-19 Ran a 20-second closed-loop eval on GPU0 `model_200.pt`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task027/unitree_mjlab_eval/gpu0_model200_forward_clean_256x1000.json`.
  Result: not passed. It had zero falls over 256 envs x 1000 steps, but
  forward velocity error was still about 0.498 m/s, above the 0.35 m/s
  threshold, so the policy was upright but not yet walking/tracking.
- 2026-05-19 Started a detached eval monitor:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/.agent/tmp/task027_start_unitree_mjlab_eval_monitor.sh`.
  Monitor pid: `510530`. Output dir:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task027/unitree_mjlab_eval_monitor`.
  The monitor starts at `model_500.pt`, evaluates every 500 iterations, runs a
  quick eval first, then full eval only if quick passes. It requires both
  seed42 and seed43 runs to pass full eval before writing `OVERALL_PASS.json`.
- 2026-05-19 Training status after monitor start: both full training processes
  still running. GPU0 reached learning iteration 314/10001 with reward around
  11 and episode length near 980. GPU1 reached learning iteration 346/10001
  with reward around 16 and episode length near 990-1000.
- 2026-05-19 Rechecked the H200 runs after the monitor had time to evaluate.
  The training processes were still alive: GPU0 reached iteration 3074/10001
  with reward about 42.2 and episode length 1000; GPU1 reached iteration
  3130/10001 with reward about 42.3 and episode length 1000. The eval monitor
  exited because it wrote
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task027/unitree_mjlab_eval_monitor/OVERALL_PASS.json`.
- 2026-05-19 Closed-loop eval pass evidence:
  - seed42 `model_500.pt` full eval:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task027/unitree_mjlab_eval_monitor/gpu0_seed42_model500_full_forward_clean.json`
    passed 256 envs x 1000 steps (20 s), zero-fall ratio 1.0, mean linear
    velocity error 0.114 m/s, mean yaw velocity error 0.078 rad/s, mean
    projected-gravity xy 0.027.
  - seed43 `model_500.pt` full eval:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task027/unitree_mjlab_eval_monitor/gpu1_seed43_model500_full_forward_clean.json`
    passed 256 envs x 1000 steps (20 s), zero-fall ratio 1.0, mean linear
    velocity error 0.099 m/s, mean yaw velocity error 0.086 rad/s, mean
    projected-gravity xy 0.033.
  Both are below the configured pass thresholds: zero-fall ratio >= 0.95,
  linear velocity error <= 0.35 m/s, yaw velocity error <= 0.35 rad/s, and
  projected-gravity xy <= 0.35.
- 2026-05-19 Stopped the two long-running training processes after pass
  evidence was captured. Process check showed no remaining PIDs for
  `506403`/`506406`, no `g1_velocity_task027_from_scratch` process, and both
  H200 GPUs at 0 MiB used.
- 2026-05-19 Final saved checkpoints before stopping:
  seed42
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity_task027_from_scratch/2026-05-19_11-21-28_gpu0_env8192_seed42_full/model_3200.pt`;
  seed43
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity_task027_from_scratch/2026-05-19_11-21-28_gpu1_env8192_seed43_full/model_3300.pt`.
- 2026-05-19 Final latest-checkpoint closed-loop eval also passed:
  seed42 `model_3200.pt` wrote
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task027/unitree_mjlab_final_eval/gpu0_seed42_model3200_full_forward_clean.json`
  with zero-fall ratio 1.0, mean linear velocity error 0.073 m/s, mean yaw
  velocity error 0.035 rad/s, mean projected-gravity xy 0.027; seed43
  `model_3300.pt` wrote
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task027/unitree_mjlab_final_eval/gpu1_seed43_model3300_full_forward_clean.json`
  with zero-fall ratio 1.0, mean linear velocity error 0.082 m/s, mean yaw
  velocity error 0.059 rad/s, mean projected-gravity xy 0.035.
- 2026-05-19 Rendered a headless MuJoCo video from seed42 `model_3200.pt`
  at fixed command `[0.5, 0.0, 0.0]` for 400 steps (8 s), 960x720. The first
  attempt failed because `MUJOCO_GL=egl` was missing; rerun with
  `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` succeeded. Remote video:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task027/unitree_mjlab_render/model3200_vx0p5/task027-g1-model3200-vx0p5-step-0.mp4`.
  Remote midframe:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task027/unitree_mjlab_render/model3200_vx0p5/task027-g1-model3200-vx0p5-midframe.png`.
  Video metadata: 960x720, 400 frames, 8.0 s, 895715 bytes. Render summary had
  no done steps for env0.

## Review

Status: passed for the upstream Unitree MJLab from-scratch closed-loop
velocity baseline.

This task intentionally uses upstream Unitree MJLab first. Rewriting the policy
or runner inside `h200_locomotion_lab` is deferred until the upstream baseline
is verified and measured.

The baseline launched, produced PPO training iterations, saved checkpoints,
passed closed-loop eval on both independent seeds, and produced a headless
render video. Current best-performing configuration was two independent
processes, one per H200 GPU, with 8192 envs each. Upstream baseline is now
verified enough to use as the reference before porting or modifying the policy
inside `h200_locomotion_lab`.
