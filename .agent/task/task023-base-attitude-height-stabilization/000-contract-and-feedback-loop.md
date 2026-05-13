# Subtask 000: Contract And Feedback Loop

## Route

- Router owns the task contract, stop rules, and evidence shape.
- Coding subagent must not start before this contract exists.
- Reviewer checks that later work follows this contract.

## Contract

Use the diagnose loop:

1. Establish source and `ankle_roll_larger_spheres` baselines.
2. Add a local fixed-controller probe with unit tests.
3. Run H200 controller matrix only after local tests and review.
4. Decide from zero-action/fixed-controller evidence, not PPO reward.

## Stop Rules

- No PPO or walking.
- No downloads.
- No source asset edits.
- No `/mnt/workspace` or `/mnt/workspace1` writes/deletes.
- No `GenesisG1SceneBackend` changes.
- H200 commands must use `run_guarded.sh`.

## Log

- 2026-05-13 Created with task023.

## Review

Status: pending.
