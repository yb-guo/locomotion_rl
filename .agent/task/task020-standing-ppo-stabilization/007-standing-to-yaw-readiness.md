# 007: Standing To Yaw Readiness

## Goal

Decide whether the standing policy is stable enough for a later yaw curriculum.

This subtask does not train walking and does not open `vx_yaw`.

## Route

1. Run only after subtask006 and subtask006b pass.
2. Probe small yaw command with `vx=0`.
3. Compare reset rate and episode length to standing eval.
4. Stop if yaw collapses standing stability.

## Acceptance

- Standing remains stable enough under small yaw.
- If not, next work stays in standing PPO.
- No vx training is started.

## Log

- 2026-05-12 Planned.
- 2026-05-12 Skipped by stop rule. Subtasks006-006c did not establish a stable
  standing policy. No yaw or `vx_yaw` probe was run.

## Review

Status: skipped because standing PPO gate failed.
