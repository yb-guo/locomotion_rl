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

## Review

Status: pending. No decision yet.
