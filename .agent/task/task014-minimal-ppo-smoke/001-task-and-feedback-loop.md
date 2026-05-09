# 001: Task And Feedback Loop

## Goal

Create the task14 guardrail and first deterministic feedback loop before PPO
implementation.

## Route

1. Start from `.agent/index.md`.
2. Confirm branch/worktree:
   - branch `codex/task014-minimal-ppo-smoke`;
   - worktree
     `../_worktrees/h200-locomotion-lab-task014-minimal-ppo-smoke`.
3. Read task013 env contract and H200 evidence.
4. Add task014 docs with stop rules and acceptance.
5. Define fast local pass/fail:
   - local full pytest must pass without torch;
   - torch-dependent tests must skip when torch is absent.
6. Define H200 pass/fail:
   - guarded H200 focused pytest;
   - guarded 3-seed PPO smoke on physical GPU 1.

## Stop Rules

- If worktree is dirty with unrelated changes, stop and inspect before editing.
- If local full pytest cannot run because torch is imported at module import
  time, fix import boundary before any H200 work.
- If H200 cannot access torch/CUDA through the guarded command, stop task014 and
  record blocker.

## Verification

- `git status --short --branch`
- local full pytest
- H200 focused pytest command recorded in task log

## Log

- 2026-05-09 Confirmed branch/worktree:
  `codex/task014-minimal-ppo-smoke` at
  `../_worktrees/h200-locomotion-lab-task014-minimal-ppo-smoke`.
- Read `.agent/index.md` and task014 stop rules before implementation.
- Used task013 env contract as upstream input:
  `G1VelocityTrackingVectorizedEnv` with 90D observation and 27D action.
- Established local loop and H200 loop:
  - local focused tests for PPO and CLI boundaries;
  - local full pytest;
  - H200 focused pytest;
  - H200 3-seed PPO smoke on physical GPU 1.

## Review

Status: passed.

- Feedback loop was sufficient to catch local no-torch import regressions before
  H200 execution.
- H200 stop rule was honored through guarded commands and project-only output.
