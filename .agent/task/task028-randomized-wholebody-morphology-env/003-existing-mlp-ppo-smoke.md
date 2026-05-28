# 003: Existing MLP PPO Smoke

## Route

Train `Unitree-G1-Gripper-Flat` with the existing MJLab/RSL-RL PPO MLP stack.
Do not change policy architecture in this slice.

## Minimal Closed Loop

Feedback loop:

1. Run a tiny train smoke on H200:
   `Unitree-G1-Gripper-Flat`, tensorboard logger, no W&B, small env count, two
   learning iterations.
2. Verify a checkpoint is saved.
3. Run a short closed-loop eval on the smoke checkpoint.

Pass:

- Training starts from scratch with the existing runner.
- Actor output dim is 31 and matches env action dim.
- Two PPO iterations complete without NaNs.
- `model_1.pt` or another smoke checkpoint is saved.
- Short eval can load the checkpoint and step the env.

Fail:

- Policy architecture has to change before the smoke can run.
- Training crashes due to obs/action shape mismatch.
- Gripper action term is ignored or missing from the actor output.

Evidence:

- H200 stdout, training log dir, checkpoint path, and eval JSON under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/mlp_ppo_smoke/`.

## Log

- 2026-05-19 Opened during diagnose audit to ensure the first learning loop is
  minimal and policy-architecture-neutral.
- 2026-05-19 First PPO smoke attempt with `VelocityOnPolicyRunner` started
  successfully and verified the actor/critic contract, but failed while saving
  checkpoint metadata. The actor was a plain MLP with input 104 and output 31;
  action terms were `body_joint_pos=29` and `gripper_joint_pos=2`. The failure
  was `KeyError: 'joint_pos'` from the velocity ONNX metadata exporter, which
  hard-codes `env.action_manager.get_term("joint_pos")`. This is a runner
  exporter compatibility issue for multi-term actions, not a policy-shape
  failure.
- 2026-05-19 Patched the new gripper task registration to use the base
  `MjlabOnPolicyRunner` instead of `VelocityOnPolicyRunner`. This only affects
  the new `g1_gripper` task package and does not change the original
  `Unitree-G1-Flat` baseline. ONNX export metadata for multi-term actions is
  deferred until it is explicitly needed.
- 2026-05-19 Re-ran the smoke through a subagent worker on H200:
  `Unitree-G1-Gripper-Flat`, GPU0, 64 envs, 2 learning iterations,
  `save_interval=1`, tensorboard logger, W&B upload disabled. Training passed,
  completed iterations 0/2 and 1/2, and saved checkpoints under:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task028_smoke/2026-05-19_15-47-36_env64_iter2_gpu0_base_runner`.
- 2026-05-19 Smoke checkpoints:
  - `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task028_smoke/2026-05-19_15-47-36_env64_iter2_gpu0_base_runner/model_0.pt`
  - `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task028_smoke/2026-05-19_15-47-36_env64_iter2_gpu0_base_runner/model_1.pt`
- 2026-05-19 Ran a checkpoint-load short eval on `model_1.pt` with strict actor
  load, 8 envs x 32 steps, clean fixed `forward_0p5` scenario. Eval completed
  with zero-fall ratio 1.0 and max done count 0. Output JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/ppo_smoke/g1_gripper_env64_iter2_model1_short_eval.json`.

## Review

Status: passed for smoke.

The existing MJLab/RSL-RL MLP stack can train the new fixed-topology gripper
task without changing policy architecture. The actor output dimension is 31 and
matches the environment action dimension. The only compatibility issue found in
this slice was ONNX metadata export in the velocity-specific runner, which
assumes a single action term named `joint_pos`. The first-pass training smoke
uses the base runner for this task; a proper multi-term ONNX metadata exporter
should be a later deployment/export slice, not a blocker for learning.

This smoke does not prove locomotion quality or convergence. It proves that
the environment, observation/action shape, PPO runner, checkpoint save, and
checkpoint load path are minimally closed.
