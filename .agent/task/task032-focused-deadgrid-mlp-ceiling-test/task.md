# Task 032: Focused Dead-Grid MLP Ceiling Test

## Route

Test whether the remaining Task031 Level A forced persistent dead-grid failures
can be improved by curriculum and motor-failure parameter tuning while keeping
the current MLP PPO policy.

This task is deliberately a ceiling test, not a promise to solve the problem.

Starting point:

`/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task030_dynamic_004_train/2026-05-21_17-35-22_005_kneehiproll_vx2p0_from5320_env8192_iter30_gpu1_seed30750/model_5349.pt`

Scope:

- Keep G1-like topology, rewards, action contract, and observation contract.
- Keep actor/action `104 -> 31`.
- Keep MLP PPO.
- Do not add explicit actor fault labels or active joint ids.
- Do not change link geometry, mass, COM, or inertia.
- Tune only motor-failure curriculum parameters and training schedule.

Known blockers from Task031:

- Forced persistent dead-grid fails at low speed for `left_hip_yaw_joint` and
  `left_hip_roll_joint`.
- High-speed failures remain concentrated on `right_knee_joint`.
- Level C arbitrary onset also fails hip pitch/yaw and right knee cases, so this
  task should not expand into arbitrary onset.

Planned slices:

1. `001-curriculum-contract.md`
   - Define focused joints, curriculum stages, and non-goals.
   - Keep pass/fail thresholds identical to Task031 Level A/B.

2. `002-h200-stage-smoke.md`
   - Patch H200 MJLab with Task032 focused weak/mixed/hard env ids.
   - Run config inspect and a short smoke train.

3. `003-focused-curriculum-train.md`
   - Train staged MLP from `model_5349.pt`.
   - First weak/partial-dead focus, then mixed, then hard only if metrics
     improve.

4. `004-eval-decision.md`
   - Evaluate clean, random persistent, forced dead-grid, and dynamic switch.
   - Decide whether MLP tuning is enough or whether the next task must add
     history/GRU/LocoFormer-style memory.

## Minimal Closed Loop

1. Register and inspect Task032 envs on H200.
2. Run a short smoke train to prove the env is trainable.
3. Run one focused curriculum attempt from `model_5349.pt`.
4. Evaluate the resulting checkpoint on the exact Task031 blocker cases.
5. Stop if forced dead-grid remains stuck on the same joints after one focused
   attempt; do not keep tuning blindly.

Acceptance:

- Smoke train completes and writes a checkpoint.
- Eval JSON reports clean, random persistent, forced dead-grid, and dynamic
  switch metrics.
- Decision is explicit:
  - `MLP tuning promising`: forced dead-grid failures shrink materially without
    regressing dynamic switch.
  - `MLP ceiling reached`: failures remain concentrated on the same joints or
    dynamic switch regresses.

Pass:

- This task passes when it produces the above decision with evidence. It does
  not require all forced dead-grid cases to pass.

Fail:

- The task changes policy architecture.
- The actor observes explicit failure state.
- Training reward is used as evidence without eval JSON.
- The task keeps adding ad hoc training stages after the first focused attempt
  without a clear metric improvement.

Evidence root:

`/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task032/`

## Log

- 2026-05-28 Opened after Task031 showed Level B passes but Level A forced
  persistent dead-grid remains unresolved.
- 2026-05-28 Implemented and tested Task032 weak-focused curriculum on H200.
  Staging smoke passed, `model_5388.pt` and early `model_5350.pt` were evaluated
  on the blocker subset. Both preserve dynamic-switch behavior, but neither
  closes forced dead-grid.

## Review

Status: partial close with negative result for the first tuning hypothesis.
Weak-focused MLP curriculum is not enough to solve the Task031 Level A forced
dead-grid blocker. Do not escalate to mixed/hard without a sharper hypothesis;
the evidence now points toward history/memory or explicit failure-conditioning
as the next serious route.
