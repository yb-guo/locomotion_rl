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

## Review

Status: planned.
