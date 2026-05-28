# 002 Stage Registration

## Route

Register task-specific H200 MJLab stages for eval-gated frozen-base StackMLP K4
curriculum.

Required stages:

- clean/unified-speed rehearsal;
- weak persistent motor failure;
- mixed weak/dead persistent failure;
- forced dead-grid focused rehearsal;
- dynamic-switch rehearsal.

Keep the runner path compatible with Task033 shared history and frozen-base
StackMLP K4.

## Log

- 2026-05-28 Planned.
- 2026-05-28 Added local Task035 stage registration artifact:
  `task035_create_eval_gated_curriculum_stages.py`.
- 2026-05-28 Patched H200 MJLab registry with:
  - `Unitree-G1-Gripper-Flat-Task035-CleanUnified-FrozenBase-Fast2p0`;
  - `Unitree-G1-Gripper-Flat-Task035-WeakPersistent-FrozenBase-Fast2p0`;
  - `Unitree-G1-Gripper-Flat-Task035-MixedPersistent-FrozenBase-Fast2p0`;
  - `Unitree-G1-Gripper-Flat-Task035-ForcedDeadGrid-FrozenBase-Fast2p0`;
  - `Unitree-G1-Gripper-Flat-Task035-DynamicSwitch-FrozenBase-Fast1p6`.
- 2026-05-28 H200 env64/iter1 smoke passed for mixed persistent stage:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task035/eval_gated_curriculum_train/035_mixed_smoke_from_model5350_env64_iter1_gpu1_seed3503601.stdout.log`.
  Smoke loaded `model_5350.pt`, resolved actor `actor_history` `540D`, critic
  `119D`, and completed one PPO iteration with `fell_over=0.0`.

## Review

Status: passed for stage registration and smoke.
