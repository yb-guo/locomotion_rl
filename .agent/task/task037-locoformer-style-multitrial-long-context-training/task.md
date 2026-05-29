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
- 2026-05-29 Completed 002 MJLab auto-reset smoke. H200 summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/mjlab_multitrial_smoke/task037_mjlab_multitrial_smoke_summary.json`
  with `pass=true`, env64 one-iter smoke, env8192 one-iter smoke, and extras
  probe `done_matches_episode_done=true`.
- 2026-05-29 Completed 003 deterministic inner reset probe. H200 summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/deterministic_inner_reset/task037_mjlab_inner_reset_probe_env64_poststep_restore.json`
  with `pass=true`, `inner_command_max_delta=0.0`,
  `inner_failure_max_delta=0.0`, fixed phase `[0.0, 1.0]`, and root z `0.8`.
- 2026-05-29 Completed 004 multi-trial eval JSON. Existing Task036 AdaptK4
  checkpoint `model_5408.pt` was evaluated on H200 with 64 envs, 360 steps,
  2.0s trials, fixed 2.0 m/s command. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/multitrial_eval/task037_adaptk4_model5408_multitrial_eval_env64.json`.
  Result: `final_trial_pass=true`, `pass=true`, final-trial fall ratio `0.0`,
  final-trial velocity error `0.635804`, `promotion_gate=final_trial`,
  `quality_claim=false`.

## Review

Status: active. 001 fake-env contract, 002 MJLab auto-reset smoke, and 003
deterministic inner-reset probe passed. 004 eval JSON contract passed with an
existing AdaptK4 checkpoint. TXL memory and long-context policy quality are not
validated yet.
