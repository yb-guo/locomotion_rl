# 002 H200 Stage Smoke

## Route

Patch H200 MJLab with Task032 env ids and prove the env can instantiate and
train briefly.

Env ids:

- `Unitree-G1-Gripper-Flat-Task032-WeakFocused-Fast2p0`
- `Unitree-G1-Gripper-Flat-Task032-MixedFocused-Fast2p0`
- `Unitree-G1-Gripper-Flat-Task032-HardFocused-Fast2p0`

Smoke requirements:

- Patch script is idempotent.
- Config inspect preserves actor/action `104 -> 31`.
- `env64`, `iter2` smoke train completes from `model_5349.pt`.

## Log

- 2026-05-28 Planned as first executable loop.
- 2026-05-28 Added and synced H200 patch/launch artifacts:
  - `.agent/task/task032-focused-deadgrid-mlp-ceiling-test/task032_create_focused_curriculum_stages.py`;
  - `.agent/task/task032-focused-deadgrid-mlp-ceiling-test/task032_launch_focused_curriculum.sh`.
- 2026-05-28 Applied H200 patch idempotently. Registered env ids:
  `Unitree-G1-Gripper-Flat-Task032-WeakFocused-Fast2p0`,
  `Unitree-G1-Gripper-Flat-Task032-MixedFocused-Fast2p0`, and
  `Unitree-G1-Gripper-Flat-Task032-HardFocused-Fast2p0`.
- 2026-05-28 H200 contract inspect for weak-focused env at `0.4` and
  `2.0 m/s` showed actor/action `104 -> 31`, critic obs `119`, and
  `forbidden_actor_terms=[]`. The reused Task031 dynamic contract wrapper marks
  `pass=false` because Task032 uses reset-time `motor_failure` instead of
  `dynamic_motor_failure`; this is expected for Task032.
  Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task032/weak_focused_contract/task031_unified_speed_contract_summary.json`.
- 2026-05-28 Smoke train from `model_5349.pt` completed:
  `NUM_ENVS=64`, `MAX_ITER=2`, `SAVE_INTERVAL=1`, `seed=3203001`,
  run name `032_weak_smoke_env64_iter2_gpu1_seed3203001`. Log:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task032/focused_curriculum_train/032_weak_smoke_env64_iter2_gpu1_seed3203001.stdout.log`.

## Review

Status: passed for staging and smoke. Task032 weak-focused env instantiates and
trains from the accepted `model_5349.pt` warm start.
