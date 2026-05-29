# Task 037: LocoFormer-Style Multi-Trial Long-Context Training

## Route

Task036 showed that swapping policy consumers alone is not enough:

- AdaptK4 is the best partial candidate, but does not pass the full deadgrid.
- GRU K4 and token K4 are rejected for the current route.
- Longer memory should not be trained until the reset/memory/eval contract
  matches LocoFormer-style in-context adaptation.

This task adds a scaled-down LocoFormer-style training contract:

```text
outer episode
  sample one latent condition:
    command, motor failure, randomization condition

  trial 0
  trial 1
  final trial

  memory is preserved across inner trial resets
  memory is cleared only at outer episode reset
```

Fixed acceptance criteria:

- `num_trials = 3` for the first implementation.
- `trial_done = fall OR trial_timeout`.
- `episode_done = trial_done AND final_trial`.
- runner-facing `done = episode_done`.
- Inner trial reset preserves actor history, recurrent memory, and TXL memory.
- Outer episode reset clears actor history, recurrent memory, and TXL memory.
- Inner trial reset keeps command, failure, and randomization condition
  unchanged.
- Inner trial reset uses deterministic standing pose.
- Inner trial reset fixes phase to the start phase.
- Inner trial reset clears env/action-manager last action.
- Fall keeps the existing fall penalty.
- Trial timeout adds no extra penalty.
- Actor does not see trial index, final-trial flag, failure id, motor scale, or
  failure mask.
- Critic also does not see trial index or final-trial flag by default.
- Trial labels and condition debug state are allowed only in `extras` and JSON
  evidence.
- Task037 target long-context horizon is `3.2s`, i.e. `160` policy steps at
  `50Hz`; full `160`-step context is for TXL-style memory, not flattened MLP.

## Planned Slices

1. `001-fake-multitrial-contract.md`
   - Build a local fake vec-env harness and deterministic contract tests.

2. `002-mjlab-multitrial-smoke.md`
   - Connect the wrapper to MJLab and run construction/PPO smoke only.

3. `003-deterministic-inner-reset-mjlab.md`
   - Implement and verify real MJLab deterministic inner reset.

4. `004-multitrial-eval-json.md`
   - Add per-trial eval JSON with final-trial pass semantics.

5. `005-txl-style-memory-consumer.md`
   - Add TXL-style long-context consumer smoke with memory reset tests.

6. `006-long-context-training-decision.md`
   - Train/evaluate and compare against Task036 AdaptK4 partial.

## Minimal Closed Loop

1. Pass fake-env contract tests for `trial_done`, `episode_done`, condition
   preservation, and history reset behavior.
2. Run MJLab env64/env8192 smoke without claiming locomotion quality.
3. Prove deterministic inner reset preserves command/failure.
4. Run multi-trial eval JSON on an existing checkpoint.
5. Only then add TXL-style policy consumer and training smoke.
6. Long-train only after reset/memory/eval semantics are mechanically correct.

## Log

- 2026-05-29 Opened after Task036 closed with no promoted checkpoint and user
  approved the multi-trial contract decisions.
- 2026-05-29 Completed 001 fake-env contract tests. H200 torch validation:
  `/mnt/workspace/users/guoyubo/agent_workspace/task037_contract_tmp_20260529`,
  `tests/test_task037_multitrial_contract.py -q` -> `2 passed in 2.48s`.

## Review

Status: active. 001 fake-env contract passed; MJLab smoke/reset and policy
quality are not validated yet.
