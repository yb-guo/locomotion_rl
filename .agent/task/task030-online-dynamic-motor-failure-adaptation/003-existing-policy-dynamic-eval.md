# 003: Existing Policy Dynamic Eval

## Route

Evaluate current task029 accepted policies under the dynamic scheduler before
training on dynamic failures.

Purpose:

- Establish the baseline failure mode.
- Separate scheduler/eval bugs from learning problems.
- Find whether MLP already adapts through current proprioception feedback.

Checkpoints:

- Start with a known lower-speed accepted checkpoint at fixed `1.2 m/s` to
  validate the eval loop.
- Then evaluate task029 accepted `Fast1p6 model_4700.pt` at fixed `1.6 m/s`.

Required cases:

- clean fixed-command eval
- persistent weak/dead eval
- dynamic single-failure eval
- dynamic switch eval using the deterministic template

Pass:

- JSON evidence exists for every case.
- The dynamic metrics identify whether failure is onset instability, poor
  recovery, or switch-specific collapse.
- No training result is accepted before this baseline is recorded.

Fail:

- Existing-policy eval is skipped.
- Only training reward is used to infer dynamic robustness.
- The eval does not distinguish transient-window metrics from post-recovery
  metrics.

## Log

- 2026-05-21 Opened.

## Review

Status: open. This subtask may pass even if the current policy fails dynamic
eval, as long as the failure mode is measured.
