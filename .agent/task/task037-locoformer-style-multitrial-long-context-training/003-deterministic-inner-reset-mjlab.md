# 003 Deterministic Inner Reset MJLab

## Route

Implement the real MJLab inner-trial reset.

Acceptance criteria:

- Fall or trial timeout resets robot to deterministic standing pose.
- Phase resets to fixed start phase.
- Env/action-manager last action is cleared.
- Command is unchanged across inner trials.
- Motor failure target, failure type, severity, and actuator force range are
  unchanged across inner trials.
- Outer episode reset resamples the latent condition.
- JSON evidence records pre/post reset state and condition preservation.

## Log

- 2026-05-29 Planned.

## Review

Status: pending.
