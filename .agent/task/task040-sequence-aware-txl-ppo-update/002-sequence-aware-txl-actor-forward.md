# 002: Sequence-Aware TXL Actor Forward

## Route

Add an explicit actor entry point for PPO update sequences so update code can
run the true-TXL memory path without borrowing the inference env cache.

## Contract

The actor sequence path must:

- accept obs groups shaped `[time, env, obs_dim]`;
- normalize actor history consistently with the base MLP model;
- allocate local per-update memory tensors shaped `[env, memory_len, token_dim]`;
- apply reset masks before the timestep that follows a done;
- update sequence debug counters;
- never increment stateless fallback counters;
- never mutate `_memory_tensors`, because those belong to inference/env rollout
  cache state.

Required smoke gate fields:

- `stateless_fallback_forward_batches == 0`
- `stateless_fallback_forward_samples == 0`
- `sequence_update_forward_batches > 0`

No quality claim is allowed:

- `quality_claim:false`
- `training_claim:false`
- `eval_claim:false`
- `reproduction_claim:false`
- `superiority_claim:false`

## Log

- 2026-05-30 Added `task040_forward_sequence` and
  `task040_get_sequence_latent` to `Task038TrueTxlMemoryModel`. The method uses
  local sequence memory and leaves the inference cache untouched.
- 2026-05-30 H200 Task040 smoke confirmed actor sequence counters:
  `sequence_update_forward_batches=1`, `sequence_update_forward_samples=16`,
  `stateless_fallback_forward_batches=0`, and
  `stateless_fallback_forward_samples=0`.

## Review

Status: implemented and verified by Task040 H200 smoke.
