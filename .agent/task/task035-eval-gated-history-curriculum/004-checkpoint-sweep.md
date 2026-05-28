# 004 Checkpoint Sweep

## Route

Evaluate non-final checkpoints from the curriculum run.

Fast gates:

- `2.0 m/s` right-knee forced dead;
- `2.0 m/s` full forced dead-grid;
- `2.0 m/s` canonical dynamic switch.

Escalate only the best candidates to representative multi-speed eval.

Selection rule:

- choose by eval score;
- reject checkpoints that regress clean or dynamic-switch gates;
- never accept a checkpoint only because it is the final iteration.

## Log

- 2026-05-28 Planned.

## Review

Status: pending. No sweep evidence yet.
