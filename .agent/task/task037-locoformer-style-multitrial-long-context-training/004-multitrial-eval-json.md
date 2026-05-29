# 004 Multi-Trial Eval JSON

## Route

Add eval that reports per-trial behavior before any TXL quality claims.

Acceptance criteria:

- Eval JSON has `trial_0`, `trial_1`, `final_trial`, and aggregate sections.
- Per-trial metrics include fall ratio, velocity error, yaw error, root z,
  gravity xy, reward, and reset reason counts.
- Final-trial pass/fail is explicit and is the default promotion gate.
- Aggregate metrics are auxiliary and cannot hide final-trial failure.
- Existing AdaptK4 partial checkpoint is evaluated first.

## Log

- 2026-05-29 Planned.

## Review

Status: pending.
