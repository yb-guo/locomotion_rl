# 001: Task And Curriculum Contract

## Goal

Define task015's boundaries before implementation.

## Route

1. Read `AGENTS.md` and `.agent/index.md`.
2. Create task015 task doc with goal, scope, non-goals, H200 protocol, stop
   rules, and acceptance.
3. Define curriculum stages without adding algorithmic complexity.
4. Require coding subagent implementation and read-only reviewer review.

## Log

- 2026-05-09 Read repo agent instructions.
- 2026-05-09 Created task015 contract:
  - one branch;
  - one worktree;
  - H200 GPU1 isolation;
  - guarded remote command requirement;
  - no downloads;
  - no render/GIF/video;
  - no `GenesisG1SceneBackend` change;
  - no `/mnt/workspace*` writes/deletes.
- 2026-05-09 Curriculum stages fixed for first runner:
  - `standing`;
  - `small_vx`;
  - `small_yaw`;
  - `small_vxyaw`.

## Review

Status: passed as Router contract.

- Contract is scoped to longer-horizon PPO smoke, not walking quality.
- Acceptance requires verification evidence and read-only review.
