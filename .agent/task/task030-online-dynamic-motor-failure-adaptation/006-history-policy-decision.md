# 006: History Policy Decision

## Route

Use the Task030 MLP results to decide whether explicit history or memory is
needed in a later task.

Do not change the policy in Task030 unless the user explicitly re-scopes it.

Decision outcomes:

- If MLP passes through `2.0 m/s`, record that current proprioceptive feedback
  is sufficient for this dynamic weak/dead motor setting.
- If MLP fails at `1.6`, `1.8`, or `2.0`, diagnose whether the failure is
  onset detection, recovery timing, switch ambiguity, or speed tracking.
- Only then propose a later task for one of:
  - observation stack, e.g. `5` or `10` frames
  - GRU/recurrent policy
  - LocoFormer-style long-context policy

Pass:

- Decision is based on dynamic eval evidence, not expectation.
- Any proposed memory policy has a concrete failure mode it is meant to fix.

Fail:

- History is added before the MLP baseline has been evaluated.
- A larger model is proposed without identifying which dynamic metric failed.

## Log

- 2026-05-21 Opened.

## Review

Status: open. This is a decision subtask, not an implementation subtask for a
new policy.
