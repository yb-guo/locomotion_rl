# 004: Minimal Standing Reward Pack

## Goal

Create a minimal standing reward that promotes active balance without walking
or reward hacking.

## Route

1. Keep command tracking disabled or neutral for standing.
2. Add/verify components:
   - upright reward;
   - base height target reward;
   - joint velocity penalty;
   - action rate penalty;
   - default pose deviation penalty;
   - small alive reward;
   - termination penalty.
3. Record component scales every update.
4. Reject reward if reward improves while survival worsens.

## Acceptance

- Reward components are finite and comparable.
- Root height does not learn permanent low crouch.
- Action saturation does not rise with reward.
- Survival or episode length improves in H200 smoke.

## Log

- 2026-05-12 Planned.
- 2026-05-12 Added opt-in `joint_velocity_penalty_scale` plumbing with default
  `0.0` to preserve existing PPO reward behavior. Env now reports raw
  `joint_velocity_penalty` as mean squared DoF velocity and subtracts
  `joint_velocity_penalty_scale * joint_velocity_penalty` when explicitly
  enabled.
- 2026-05-12 Forwarded `--joint-velocity-penalty-scale` through `g1_ppo_smoke`
  and curriculum smoke config compatibility. PPO reward component aggregation
  and smoke contribution stats now include
  `reward_contribution_joint_velocity_penalty_mean` with negative penalty sign.
- 2026-05-12 Local verification:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_g1_velocity_tracking_env.py tests\test_g1_ppo_smoke.py tests\test_ppo_loop.py -q -p no:cacheprovider`
  -> 22 passed, 6 skipped.
- 2026-05-12 Compatibility verification:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_g1_curriculum_ppo_smoke.py -q -p no:cacheprovider`
  -> 10 passed, 4 skipped.
- 2026-05-12 Router local extended verification:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_g1_velocity_tracking_env.py
  tests\test_g1_curriculum_ppo_smoke.py tests\test_g1_policy_action_safety_probe.py
  tests\test_g1_ppo_smoke.py tests\test_ppo_loop.py -q -p no:cacheprovider`
  -> 34 passed, 10 skipped.
- 2026-05-12 H200 focused verification through guarded command:
  `PYTHONPATH=src python -m pytest tests/test_g1_velocity_tracking_env.py
  tests/test_g1_ppo_smoke.py tests/test_ppo_loop.py tests/test_g1_curriculum_ppo_smoke.py
  -q -p no:cacheprovider` -> 42 passed in 2.89s.
- 2026-05-12 H200 extended compatibility verification through guarded command:
  `PYTHONPATH=src python -m pytest tests/test_g1_velocity_tracking_env.py
  tests/test_g1_curriculum_ppo_smoke.py tests/test_g1_policy_action_safety_probe.py
  tests/test_g1_ppo_smoke.py tests/test_ppo_loop.py -q -p no:cacheprovider`
  -> 44 passed in 2.99s.
- 2026-05-12 H200 reward-pack run through guarded command:
  `CUDA_VISIBLE_DEVICES=1`, physical GPU 1, logical `cuda:0`,
  `command_mode=standing`, `action_scale_mult=0.25`, `root_z=1.20`,
  `termination_height_min=0.20`, `base_height_reward_scale=0.20`,
  `joint_velocity_penalty_scale=0.001`, `termination_penalty=-1.0`, run id
  `h200-gpu1-standing-reward-pack-v1`. Result: status ok, 3/3 seeds passed,
  min collect throughput 16598.49 env-policy steps/s, mean final reward
  1.82327, mean final episode_length_mean 51.9222, mean final survival_rate
  1.0, max final height_reset_rate 0.0, max final tilt_reset_rate 0.0,
  max final timeout_rate 0.0, no final full-env reset wave. Final action
  saturation ratio stayed about 0.0026. Final root_height_mean stayed about
  0.779, root_height_min ranged about 0.543-0.601. Base-height reward
  contribution was about +0.190, joint-velocity penalty contribution about
  -0.00005.
- 2026-05-12 Decision: the minimal reward pack increases reward without
  worsening survival, reset causes, root-height profile, or action saturation.
  It is acceptable as the fixed reward config for subtask005 action-energy
  ablation.
- 2026-05-12 Pulled small H200 evidence files to local ignored task artifacts:
  `.agent/task/task020-standing-ppo-stabilization/artifacts/h200-gpu1-standing-reward-pack-v1/`
  contains `config.json`, `summary.json`, and `metrics.jsonl`. Checkpoints were
  not pulled.

## Review

- 2026-05-12 Read-only reviewer found no blocking implementation issue. Initial
  evidence gap was H200-only; the H200 run above resolves subtask004 acceptance.
- 2026-05-12 Re-review found no blocking findings. Reviewer agreed H200
  evidence satisfies subtask004 acceptance and noted remaining risk is task-level
  only: this is still short smoke, not deterministic standing eval or final gate.

Status: complete.
