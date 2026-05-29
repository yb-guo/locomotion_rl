# 001 Fake Multi-Trial Contract

## Route

Build the contract before touching MJLab.

Use a fake vectorized env with scripted raw done events, reset counters, and
condition ids. The fake env must make failure obvious if the wrapper clears
history on inner trial reset or changes the latent condition.

Acceptance criteria:

- `trial_done=True` maps to runner-facing `done=False` until final trial.
- `episode_done=True` maps to runner-facing `done=True`.
- `trial_done = fall OR trial_timeout`.
- `episode_done = trial_done AND final_trial`.
- Independent vectorized envs maintain independent trial counters.
- Inner trial reset preserves history/memory.
- Outer episode reset clears history/memory.
- Command/failure/randomization condition id is unchanged across inner trials.
- Inner reset clears last action and appends post-reset obs with zero action.
- Actor/critic observations do not contain trial index or final-trial flag.
- Trial labels are emitted only through `extras`.

This subtask does not evaluate walking quality and does not require H200.

## Log

- 2026-05-29 Planned.

## Review

Status: pending.
