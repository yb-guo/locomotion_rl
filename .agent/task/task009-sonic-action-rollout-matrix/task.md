# Task 009: SONIC Action Rollout Matrix

## Goal

Run profile-backed SONIC online action rollouts on H200 for multiple commanded
motion styles, then record numeric and visual evidence.

This task starts from task008's runtime profile foundation. It must exercise:

```text
SONIC planner -> encoder -> decoder -> profile-backed action bridge
  -> Genesis G1 robot execution
```

## Scope

- Re-run the official-context walking route after task008 migration.
- Add at least one dance-like or non-walking command sweep that changes command
  direction/facing/velocity enough to stress the action bridge and robot
  execution path.
- Generate logs and GIF/MP4 where practical.
- Record pass/fail metrics and artifact paths.

## Non-Goals

- No real Unitree G1 hardware command publishing.
- No new checkpoints, datasets, assets, or upstream repos.
- No training loop.
- No claim that a dance-like command is an official SONIC dance skill unless
  official assets prove it.

## Subtasks

- `001-route-and-assets.md`
- `002-walking-official-context.md`
- `003-dance-like-command-sweep.md`
- `004-summary-review.md`

## Acceptance

- H200 commands run through `/root/agent_workspace/safe_agent/run_guarded.sh`.
- Walking rollout records finite encoder tokens, decoder obs/actions, stable
  height, locomotion/contact metrics, and visual artifact path.
- Dance-like rollout records either a pass with summary/GIF or a concrete
  failure mode with exact command and logs.
- Each subtask has `Route / Log / Review`.

## Review

Status: opened.
