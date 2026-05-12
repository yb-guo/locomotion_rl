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

## Review

Status: local passed; H200 standing PPO reward-pack acceptance remains pending
router-run evidence with explicit reward scales.
