# 003 Focused Curriculum Train

## Route

Run one focused MLP curriculum attempt from `model_5349.pt`.

Initial training:

- Start with `Task032-WeakFocused-Fast2p0`.
- Use `8192` envs if GPU memory permits.
- Keep learning rate conservative to avoid destroying dynamic-switch behavior.
- Save frequent checkpoints for early eval.

Escalation:

- Move to mixed/hard only if weak-focused improves forced dead-grid eval or at
  least does not regress clean/dynamic behavior.

## Log

- 2026-05-28 Planned after H200 stage smoke.
- 2026-05-28 Ran weak-focused curriculum from `model_5349.pt`:
  `NUM_ENVS=8192`, `MAX_ITER=40`, `SAVE_INTERVAL=5`, `seed=3203203`,
  run name `032_weak_focused_from5349_env8192_iter40_gpu1_seed3203203`.
  Training completed in about 68 seconds at roughly `117k-118k steps/s` with no
  fell-over terminations in the final logged iterations. Log:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task032/focused_curriculum_train/032_weak_focused_from5349_env8192_iter40_gpu1_seed3203203.stdout.log`.
- 2026-05-28 Checkpoints:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task032_mlp_ceiling_train/2026-05-28_10-18-24_032_weak_focused_from5349_env8192_iter40_gpu1_seed3203203/model_5350.pt`
  through
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task032_mlp_ceiling_train/2026-05-28_10-18-24_032_weak_focused_from5349_env8192_iter40_gpu1_seed3203203/model_5388.pt`.

## Review

Status: completed for the weak-focused stage. Do not continue to mixed/hard
unless eval shows material improvement over Task031 baseline.
