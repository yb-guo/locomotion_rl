# 001: Contract Standing PPO Only

## Goal

Lock task020 scope to standing PPO active balance.

## Route

1. Freeze non-goals: no walking, no `vx_yaw`, no asset/importer/MuJoCo route.
2. Define H200 protocol and GPU isolation.
3. Define subagent workflow: coding worker implements, read-only reviewer
   reviews.
4. Define stop rules for each later subtask.

## Acceptance

- Task docs prevent task019-style drift.
- `command_mode=standing` remains mandatory until the standing gate passes.
- Task cannot pass without H200 evidence and review.

## Log

- 2026-05-12 Planned.
- 2026-05-12 Scope locked to current Genesis G1 27DoF no-hand standing PPO.
  The task020 branch was updated with master task015-019 history before
  execution. No walking, `vx_yaw`, asset/importer, MuJoCo, or zero-action proof
  route is opened.

## Review

Status: route accepted; read-only review pending after implementation evidence.
