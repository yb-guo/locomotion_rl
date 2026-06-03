# 001: RSL Storage and Update Contract

## Route

Define the minimum fix for the Task039 stateless fallback diagnosis by reading
the MJLab/RSL-RL runner, storage, and PPO update boundary.

## Contract

Current RSL-RL feed-forward update uses:

- rollout storage tensors shaped `[time, env, ...]`;
- `mini_batch_generator`, which flattens `[time, env]` to `[time * env]`;
- shuffled minibatches passed to the actor as ordinary MLP batches.

For `Task038TrueTxlMemoryModel`, that flat update batch is not a valid temporal
cache stream. It changes batch size relative to the env cache and triggers the
actor's stateless fallback.

Task040 requires a separate update path:

- preserve rollout order as `[time, env]`;
- split only along env axis when minibatching;
- derive reset masks from previous-step `dones`;
- call a sequence-specific actor API;
- keep the existing inference env cache separate;
- accept only smoke evidence where `stateless_fallback_forward_batches == 0`
  and `sequence_update_forward_batches > 0`.

No quality claim is allowed:

- `quality_claim:false`
- `training_claim:false`
- `eval_claim:false`
- `reproduction_claim:false`
- `superiority_claim:false`

## Log

- 2026-05-30 H200 source inspection found `RolloutStorage.mini_batch_generator`
  flattening time/env and `PPO.update` using that flat generator for
  non-recurrent policies. This explains Task039's fallback evidence.

## Review

Status: contract defined. Verification belongs to `003`.
