# Task 040: Sequence-Aware True-TXL PPO Update

## Route

Task039 proved that the Task038 true-TXL runner can execute, but its PPO update
path still flattens rollout samples into feed-forward minibatches. That shape
mismatch triggers `Task038TrueTxlMemoryModel`'s stateless minibatch fallback, so
the current training path cannot support a long-memory training claim.

This task fixes that boundary first. It does not attempt full LocoFormer
reproduction, policy-quality improvement, speed curriculum, or held-out
morphology.

## Fixed Scope

- Keep the Task038 G1-like true-TXL runner and obs/action/env contracts.
- Keep the inference env-cache fallback for safety.
- Add a separate sequence-aware PPO update path used only by Task040 smoke.
- Preserve rollout order as `[time, env]` during actor update.
- Use the current MLP critic path unless a later diagnostic proves it must be
  sequence-aware too.
- Do not touch `.test_tmp_task021/`.

Out of scope:

- quality, training-success, eval-success, reproduction, or superiority claims;
- reward/action/obs contract changes;
- speed/morphology/held-out expansion;
- downloading checkpoints, datasets, simulator assets, or upstream repos.

## Planned Slices

1. `001-rsl-storage-and-update-contract.md`
   - Inspect the RSL-RL storage/update boundary and define the minimum sequence
     update contract.

2. `002-sequence-aware-txl-actor-forward.md`
   - Add an explicit true-TXL sequence forward path that does not use or mutate
     the inference env cache.

3. `003-sequence-aware-ppo-update-smoke.md`
   - Add a Task040 PPO subclass and smoke CLI that proves one PPO update can
     run with `stateless_fallback_forward_batches == 0` and
     `sequence_update_forward_batches > 0`.

## Acceptance Criteria

Task040 is accepted only when:

- local tests cover the Task040 CLI pass/fail gates and no-overclaim contract;
- `python -m h200_locomotion_lab.tools.task040_sequence_txl_ppo_update_smoke --help`
  works;
- H200 smoke writes a JSON summary with:
  - `pass=true`;
  - `algorithm_class=Task040SequenceAwareTrueTxlPPO`;
  - `runner_cls=Task038TrueTxlMemoryK160Runner`;
  - `actor_model_class=Task038TrueTxlMemoryModel`;
  - `stateless_fallback_forward_batches == 0`;
  - `stateless_fallback_forward_samples == 0`;
  - `sequence_update_forward_batches > 0`;
  - `algorithm_debug.sequence_update_batches > 0`;
  - finite loss fields in `algorithm_debug.last_loss_dict`;
  - `quality_claim:false`;
  - `training_claim:false`;
  - `eval_claim:false`;
  - `reproduction_claim:false`;
  - `superiority_claim:false`.

## Evidence Gate

Local commands:

```powershell
$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task040_sequence_txl_ppo_update_smoke --help
$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task040_sequence_txl_ppo_update_smoke.py tests\test_task038_true_txl_inference_cache_safety.py tests\test_agent_inventory.py
python -m h200_locomotion_lab.tools.inspect_agent
```

H200 command shape:

```bash
PYTHONPATH=src python -m h200_locomotion_lab.tools.task040_sequence_txl_ppo_update_smoke \
  --output-json /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task040/sequence_txl_ppo_update_smoke/sequence_txl_ppo_update_env8_steps2_iter1.json \
  --log-dir /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task040/sequence_txl_ppo_update_smoke/logs \
  --num-envs 8 \
  --rollout-steps 2 \
  --iterations 1 \
  --num-mini-batches 1 \
  --device cuda:0
```

## Log

- 2026-05-30 Opened from Task039 decision
  `sequence_aware_txl_ppo_update_required_next`.
- 2026-05-30 Implemented local Task040 sequence-aware update plumbing:
  `Task038TrueTxlMemoryModel.task040_forward_sequence`,
  `Task040SequenceAwareTrueTxlPPO`, and
  `h200_locomotion_lab.tools.task040_sequence_txl_ppo_update_smoke`.
- 2026-05-30 Local verification:
  `$env:PYTHONPATH='src'; $env:PYTHONDONTWRITEBYTECODE='1'; python -m h200_locomotion_lab.tools.task040_sequence_txl_ppo_update_smoke --help`
  passed, and focused regression returned `13 passed in 0.12s` for
  `tests\test_task040_sequence_txl_ppo_update_smoke.py`.
- 2026-05-30 H200 smoke passed. JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task040/sequence_txl_ppo_update_smoke/sequence_txl_ppo_update_env8_steps2_iter1.json`.
  Result: `pass=true`, `algorithm_class=Task040SequenceAwareTrueTxlPPO`,
  `runner_cls=Task038TrueTxlMemoryK160Runner`,
  `actor_model_class=Task038TrueTxlMemoryModel`, `action_dim=31`,
  `learn_returned=true`, `stateless_fallback_forward_batches=0`,
  `stateless_fallback_forward_samples=0`,
  `sequence_update_forward_batches=1`,
  `sequence_update_forward_samples=16`,
  `algorithm_debug.sequence_update_batches=1`,
  `algorithm_debug.sequence_update_samples=16`, finite loss fields
  `value=0.01634225621819496`, `surrogate=0.31462955474853516`,
  `entropy=43.987091064453125`, and all no-overclaim flags false.

## Review

Status: Task040 minimum sequence-aware PPO update smoke is implemented and
verified. This closes only the update-boundary plumbing. It is not a
policy-quality, training-success, eval-success, reproduction, or superiority
claim.
