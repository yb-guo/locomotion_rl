# Task 055: Hidden Online Motor Process

## Route

Keep motor events in the environment/critic boundary, never in actor
observation.  Preserve events across `trial_done`, resample and restore at
`context_done`, and expose strength, delay, and EMA traces only as privileged
diagnostics.

## Log

- 2026-08-19: Added seeded weak/dead/latency event scheduling, lower-body 70%
  preference, recovery/persistent semantics, and action processing with hidden
  strength/delay/EMA.

## Review

The deterministic artifact `motor_trace_55001.json` shows a scheduled dead
event at the exact onset step, preserves its event through a trial trace, and
exposes only `strength`, `extra_latency_steps`, and `ema_alpha` in the critic
payload.  Clean-gait degradation and 10-second quality gates remain pending
on the RTX 5060 Ti long-run path.
