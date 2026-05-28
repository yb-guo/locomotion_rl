# 006 Decision

## Route

Decide whether eval-gated curriculum training improves the current route.

Decision options:

- `curriculum_promote`: selected curriculum checkpoint beats or matches
  `model_5350.pt` and videos are acceptable;
- `model5350_promote`: `model_5350.pt` validates, but curriculum does not beat
  it;
- `candidate_only`: useful evidence exists, but validation holes remain;
- `reject_route`: `model_5350.pt` does not reproduce or curriculum checkpoints
  regress badly.

## Log

- 2026-05-28 Planned.
- 2026-05-28 Decision: `candidate_only`.
  - `model_5350.pt` is validated for the scoped `2.0 m/s` gate, but not as a
    full-speed main checkpoint.
  - Task035 mixed curriculum produces checkpoints that pass fast gates and
    `1.2/2.0 m/s` full validation, but `model_5369.pt` regresses `0.4 m/s`
    dead-grid versus `model_5350.pt` (`7/12` vs `9/12`).
  - Do not promote the Task035 curriculum checkpoint.
- 2026-05-28 Reopened as Task035 subtask 007. The follow-up low-speed repair
  evidence is useful but should be treated as part of curriculum design, not as
  a separate promoted task.

## Review

Status: reopened with subtask 007. The previous checkpoint remains
non-promotable; the next step is multi-objective curriculum, not a blind
continuation of the same mixed persistent stage.
