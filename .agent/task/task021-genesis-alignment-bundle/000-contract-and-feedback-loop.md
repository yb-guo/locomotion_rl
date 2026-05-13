# Subtask 000: Contract And Feedback Loop

## Route

- Define the parameter families the alignment report must cover.
- Keep the first feedback loop local and deterministic.
- Require explicit `missing` records for unrepresented contact/friction/solver
  semantics.
- Keep PPO, rendering, asset downloads, and importer work out of this task.

## Log

- 2026-05-12 Contract created. The core diagnostic signal is a JSON alignment
  report, not a training run.

## Review

Status: router-reviewed. This subtask is a planning/contract unit; code review
starts with subtask001.
