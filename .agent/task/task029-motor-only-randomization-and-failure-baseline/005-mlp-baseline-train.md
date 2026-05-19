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

## Review

Status: pending.

The acceptance criterion is not perfect dead-motor robustness. The first pass
should show that the MLP remains trainable and can walk under the configured
episode-start persistent weak/dead leg motor distribution.
