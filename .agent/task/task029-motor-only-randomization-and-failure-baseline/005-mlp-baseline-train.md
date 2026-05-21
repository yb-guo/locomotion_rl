# 005: MLP Baseline Train

## Route

Train the existing MJLab/RSL-RL PPO MLP policy on the task029 motor-only and
persistent motor-failure stage. This is a baseline convergence task, not a
policy architecture task.

Training contract:

- Use the task028 G1-like whole-body gripper env family as the base.
- Preserve actor obs/action contract.
- Use existing MLP PPO first.
- Train on motor-only randomization plus episode-start persistent weak/dead leg
  motor sampling.
- Keep link/contact/sensor randomization disabled.

## Minimal Closed Loop

Feedback loop:

1. Start with a short 64-env or 256-env training smoke.
2. Scale toward the task027/task028 H200 env-count pattern only after smoke.
3. Track clean eval and motor-randomized eval periodically from saved
   checkpoints.
4. Stop early only with a saved checkpoint and JSON evidence.

Pass:

- Training launches on H200 and uses the intended GPU/env-count configuration.
- A saved checkpoint can be loaded for deterministic eval.
- Clean eval does not clearly regress against the task028 MLP baseline.
- Motor-only randomized eval shows stable walking under the first acceptance
  failure distribution.

Fail:

- Training requires changing actor obs or action contract.
- Training silently enables forbidden randomization.
- Checkpoints cannot be loaded for eval.
- The run only reports training reward without closed-loop eval evidence.

Evidence:

- Planned output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/mlp_baseline_train/`.

## Log

- 2026-05-19 Opened as the first full-training gate for task029. The purpose is
  to determine whether the existing MLP can learn a robust gait before any
  LocoFormer-style policy work begins.
- 2026-05-19 Launched H200 single-GPU PPO MLP baseline training from scratch on
  `Unitree-G1-Gripper-Flat-MotorOnly-Failure`, PID `601862`.
  Command:
  `/mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python
  scripts/train.py Unitree-G1-Gripper-Flat-MotorOnly-Failure --gpu-ids=[1]
  --env.scene.num-envs=8192 --agent.max-iterations=1200
  --agent.save-interval=100
  --agent.experiment-name=g1_gripper_velocity_task029_mlp_baseline_005_train
  --agent.run-name=005_failure_env8192_iter1200_gpu1_seed29005
  --agent.logger=tensorboard --agent.upload-model=False --agent.seed=29005`.
  Runtime env used `PYTHONPATH=/tmp/task029_ipython_stub:/tmp/task029_pydeps:.`,
  `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and `WANDB_DISABLED=true`.
  Log dir:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_mlp_baseline_005_train/2026-05-19_20-45-49_005_failure_env8192_iter1200_gpu1_seed29005`.
  Stdout mirror:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/mlp_baseline_train/005_failure_env8192_iter1200_gpu1_seed29005.stdout.log`.
- 2026-05-19 Verified training artifacts at the H200 log dir: `params/env.yaml`,
  `params/agent.yaml`, TensorBoard event file, and checkpoints
  `model_0.pt`, `model_100.pt`, `model_200.pt`, `model_300.pt`,
  `model_400.pt`, `model_500.pt`, and `model_600.pt`. The `model_600.pt`
  checkpoint path is
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_mlp_baseline_005_train/2026-05-19_20-45-49_005_failure_env8192_iter1200_gpu1_seed29005/model_600.pt`.
- 2026-05-19 At `model_600.pt` evidence collection, PID `601862` was still
  running at about 19:50 elapsed on GPU1 with 8192 envs, GPU memory about
  6689 MiB, GPU utilization about 84%, and recent throughput about
  111k steps/s. Stdout tail around iterations 598-603 showed mean reward
  around 27-29, mean episode length around 923-960, `time_out` around
  5.3-7.4, and `fell_over` around 0.25-0.63.
- 2026-05-19 Ran a minimal checkpoint-load probe for `model_600.pt` on GPU0
  with 16 envs and one policy step. JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/mlp_baseline_train/model_600_checkpoint_load_probe.json`.
  The probe passed with task id `Unitree-G1-Gripper-Flat-MotorOnly-Failure`,
  `actor_obs_dim=104`, `critic_obs_dim=119`, `action_dim=31`, action shape
  `[16, 31]`, reset events limited to `reset_base`, `reset_robot_joints`, and
  `motor_failure`, `forbidden_randomization_active=false`, and zero dones after
  one step. An earlier identical `model_200.pt` load probe also passed at
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/mlp_baseline_train/model_200_checkpoint_load_probe.json`.
- 2026-05-19 Training completed and PID `601862` exited. The final saved
  checkpoint is `model_1199.pt`, not `model_1200.pt`, because the run's final
  printed iteration was `1199/1200`. Final checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_mlp_baseline_005_train/2026-05-19_20-45-49_005_failure_env8192_iter1200_gpu1_seed29005/model_1199.pt`.
  The final log dir also contains `model_0.pt`, every 100-iteration checkpoint
  from `model_100.pt` through `model_1100.pt`, `params/env.yaml`,
  `params/agent.yaml`, and TensorBoard event file
  `events.out.tfevents.1779194771.nb-8drvv1aanb-0.601862.0`.
- 2026-05-19 Final stdout tail at iteration `1199/1200` reported
  `steps/s=113901`, mean reward `25.35`, mean episode length `970.25`,
  mean action std `0.43`, `Episode_Termination/time_out=5.4583`,
  `Episode_Termination/fell_over=0.3333`, `Metrics/twist/error_vel_xy=1.8284`,
  and `Metrics/twist/error_vel_yaw=1.0961`. GPU1 had returned to idle
  (`4 MiB`, `0%`) after process exit.
- 2026-05-19 Final 006 ran on `model_1199.pt` and completed the clean,
  motor-primitives, in-distribution failure, doubled holdout, and dead-motor
  grid cases. The aggregate failed the left-knee-dead grid gate:
  `dead_motor_grid_05_left_knee_joint`. Training completed, but it does not
  establish accepted robustness for task029.

## Review

Status: training complete; accepted robustness not established.

The acceptance criterion is not perfect dead-motor robustness. The first pass
should show that the MLP remains trainable and can walk under the configured
episode-start persistent weak/dead leg motor distribution.

Training completed on H200 with the intended task id, GPU/env-count, checkpoint
cadence, TensorBoard logging, no W&B upload, YAML configs, and loadable
checkpoint evidence through `model_600.pt`. The final available checkpoint for
downstream eval is `model_1199.pt`. The final 006 closed-loop eval ran on
`model_1199.pt` and failed only the left-knee-dead grid gate
`dead_motor_grid_05_left_knee_joint`, so this training run does not establish
accepted task029 robustness.
