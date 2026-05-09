# 005: Review And Decision

## Goal

Review evidence and decide the next engineering direction.

## Route

1. Read-only reviewer checks:
   - boundary compliance;
   - ablation correctness;
   - artifact completeness;
   - H200 evidence.
2. Fix blocking findings through coding subagent.
3. Record final diagnosis decision.

## Log

- 2026-05-09 Read-only reviewer found two blocking issues before H200:
  baseline mismatch did not stop the matrix, and failed `run_smoke` summaries
  could be reported as completed.
- 2026-05-09 Router fixed both issues and added regression coverage.
- 2026-05-09 H200 v1 found the Genesis singleton issue for same-process
  variant execution.
- 2026-05-09 Router fixed variant execution by using one subprocess per
  variant and re-ran local plus H200 focused tests.
- 2026-05-09 H200 v2 produced complete evidence for all four variants.
- 2026-05-09 Final read-only reviewer found no blocking findings.

Decision:

The evidence is inconclusive for a single positive cause, but it rules out the
first three task016 variants as sufficient fixes:

- Lower LR reduces KL but does not materially reduce reset waves.
- Stronger termination penalty does not prevent tilt resets.
- Stronger action-rate penalty does not prevent tilt resets.

The most supported current interpretation is that the reset wave is not mainly
a PPO schedule issue, not a missing scalar termination penalty, and not solved
by action-rate smoothing alone. Because tilt appears at update 2 even in
`standing`, with height termination absent, the next diagnostic should target
action amplitude/control semantics and upright/joint-deviation shaping before
larger curriculum changes.

Recommended next subtask:

- Compare lower `action_scale_mult` values, restricted action joint groups, and
  stronger upright/joint-deviation penalties on the same seed-0 standing-first
  loop.
- Keep baseline reproduction and one-variable stop rules from task016.

## Review

Status: passed.

- Final read-only review found no blocking findings.
- Residual risk: subprocess path is mainly covered by H200 v2 evidence rather
  than a direct local subprocess integration test.
