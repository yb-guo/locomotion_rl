# 004: Randomization Curriculum

## Route

Add randomization in stages after the no-randomization MLP smoke is learnable.
One variable group changes at a time so failures are diagnosable.

Initial order:

1. No randomization control.
2. Motor strength / PD gain scale.
3. Contact friction.
4. Link mass / COM / inertia.
5. Encoder bias / observation noise.
6. Action delay / smoothing.
7. Combined randomization.

## Minimal Closed Loop

Feedback loop:

1. For each randomization stage, run a fixed short PPO budget from scratch.
2. Run deterministic eval and randomized holdout eval on the same checkpoint.
3. Compare against the previous stage using the same JSON metrics.

Pass:

- Each stage can be toggled independently from config.
- No-randomization control remains reproducible.
- The new stage does not break import, reset, or smoke training.
- If a stage fails, the failing stage is isolated to one randomization group.

Fail:

- Multiple new randomization groups are enabled at once.
- There is no deterministic control eval.
- Failure cannot be attributed to a single stage.
- Randomization changes topology, DoF, action dim, or observation dim.

Evidence:

- Per-stage train/eval summaries under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/randomization_curriculum/`.

## Log

- 2026-05-19 Opened during diagnose audit to prevent all-randomization-at-once
  failures.

## Review

Status: planned.
