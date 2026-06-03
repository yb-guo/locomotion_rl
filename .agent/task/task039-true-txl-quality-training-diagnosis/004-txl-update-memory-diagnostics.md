# 004 TXL Update Memory Diagnostics

## Route

Diagnose whether the current Task038 true-TXL PPO update path actually trains
long memory.

Task038 deliberately allowed a stateless fallback for flattened PPO minibatches
so the smoke could run. Task039 must measure whether that fallback prevents a
real long-memory training claim.

## Minimal Closed Loop

Close this slice with a training/update diagnostic JSON that reports:

- total actor forward batches;
- env-cache stateful forward batches;
- stateless fallback forward batches;
- fallback sample count;
- memory length before/after update where observable;
- whether minibatches preserve temporal segment structure;
- a router decision on whether sequence-aware TXL PPO update is required next.

## Evidence Gate

Evidence must include:

- focused local tests for any new debug summary helper;
- one H200 diagnostic run using the same true-TXL runner path as `003`;
- explicit thresholds or interpretation:
  - high stateless fallback means no long-memory training claim;
  - low/no fallback with finite training means long-memory update path is
    plausible but still needs quality evidence.

## Subagent Ownership

Worker owns only debug counters, diagnostic CLI/logging, tests, and this doc.
Worker must not rewrite PPO sequence batching in this slice unless the router
opens a follow-up implementation slice.

Reviewer checks that the diagnostic distinguishes inference memory from training
memory and that it does not confuse either with quality.

## Failure Exit

If debug counters cannot tell whether PPO update used memory, stop and add a
lower-level instrumentation seam before running more long training.

## Log

- 2026-05-30 Opened because Task038 `013` used stateless fallback for flattened
  PPO update minibatches.
- 2026-05-30 Implemented local Task039/004 diagnostic plumbing without running
  H200:
  - `Task038TrueTxlMemoryModel.txl_debug_snapshot()` now exposes actor forward
    counters for total batches/samples, env-cache stateful batches/samples, and
    stateless fallback batches/samples. This is instrumentation only and does
    not change PPO update semantics.
  - Added
    `src/h200_locomotion_lab/tools/task039_txl_update_memory_diagnostics.py`.
    It wraps the train-only Task038 PPO update smoke, rejects held-out tasks,
    keeps all no-overclaim flags false, reports fallback ratios by batches and
    samples, records whether memory before/after lengths are observable, and
    separates `pass` as diagnostic-evidence validity from
    `long_memory_training_claim_supported`.
  - Because the wrapped Task038 smoke exposes only a post-learn `txl_debug`
    snapshot, local output explicitly sets `memory_lengths_observable=false`
    unless future evidence provides both before-update and after-update memory
    snapshots.
  - When stateless fallback is present, the wrapper reports
    `minibatches_preserve_temporal_segments=false`,
    `long_memory_training_claim_supported=false`, and router decision
    `sequence_aware_txl_ppo_update_required_next`.
  - Added `tests/test_task039_txl_update_memory_diagnostics.py` for parse/help,
    preflight, pure summary ratios, missing-counter rejection, no-overclaim
    flags, train-only delegation, and structured preflight failure.
- 2026-05-30 Router local focused verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp
  .test_tmp_task039_004_router
  tests\test_task039_txl_update_memory_diagnostics.py
  tests\test_task038_true_txl_inference_cache_safety.py
  tests\test_task038_true_txl_ppo_update_smoke.py
  tests\test_agent_inventory.py` returned `33 passed in 0.28s`.
- 2026-05-30 Ran the H200 diagnostic on the same true-TXL runner path as `003`:
  - task/config id:
    `Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke`;
  - runner class: `Task038TrueTxlMemoryK160Runner`;
  - actor model class: `Task038TrueTxlMemoryModel`;
  - action dimension: `31`;
  - envs: `8`;
  - rollout steps: `2`;
  - iterations: `1`;
  - seed: `3900401`;
  - JSON:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/txl_update_memory_diagnostics/true_txl_ppo_update_env8_steps2_iter1_memory_diagnostic.json`.
- 2026-05-30 H200 diagnostic result:
  - `pass=true` means diagnostic-evidence validity only;
  - `diagnostic_evidence_valid=true`;
  - `task038_ppo_update_smoke_pass=true`;
  - `total_actor_forward_batches=3`;
  - `total_actor_forward_samples=32`;
  - `env_cache_stateful_forward_batches=2`;
  - `env_cache_stateful_forward_samples=16`;
  - `stateless_fallback_forward_batches=1`;
  - `stateless_fallback_forward_samples=16`;
  - `stateless_fallback_ratio_by_batches=0.3333333333333333`;
  - `stateless_fallback_ratio_by_samples=0.5`;
  - `memory_lengths_observable=false` because the wrapped Task038 smoke exposes
    only a post-learn `txl_debug` snapshot;
  - `memory_lengths_after_update.mean=32.0`, `min=32`, `max=32`;
  - `minibatches_preserve_temporal_segments=false`;
  - `long_memory_training_claim_supported=false`;
  - router decision:
    `sequence_aware_txl_ppo_update_required_next`.
  - All diagnostic no-overclaim flags remained false:
    `quality_claim`, `training_claim`, `eval_claim`, `reproduction_claim`,
    and `superiority_claim`.

## H200 Evidence

The H200 diagnostic JSON is:

```text
/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task039/txl_update_memory_diagnostics/true_txl_ppo_update_env8_steps2_iter1_memory_diagnostic.json
```

This records diagnostic-evidence validity only. It makes no quality,
training-success, eval-success, reproduction, or superiority claim.

## Review

Status: closed with independent read-only review. Reviewer found no blockers.
The diagnostic supports the router decision that sequence-aware TXL PPO update
is required before a long-memory training claim.
