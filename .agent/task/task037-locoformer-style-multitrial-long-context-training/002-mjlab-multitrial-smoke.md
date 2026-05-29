# 002 MJLab Multi-Trial Smoke

## Route

Connect the multi-trial wrapper to MJLab without making real deterministic
inner reset a blocker yet.

Acceptance criteria:

- Task ids register for at least one existing consumer.
- env64 construction smoke runs.
- env8192 one PPO iteration runs.
- `extras` include `trial_done`, `episode_done`, `trial_index`,
  `final_trial`, and reset reason.
- runner-facing `done` only follows `episode_done`.
- No policy-quality claim is made from this subtask.

## Log

- 2026-05-29 Planned.

## Review

Status: pending.
