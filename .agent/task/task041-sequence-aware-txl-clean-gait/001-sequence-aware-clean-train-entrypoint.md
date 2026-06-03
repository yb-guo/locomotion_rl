# 001: Sequence-Aware Clean Train Entrypoint

## Route

Provide a repeatable train command for Task041 that uses the Task040
sequence-aware PPO algorithm for real clean-gait training, not only a smoke.

## Contract

The train CLI must:

- use `Task040SequenceAwareTrueTxlPPO`;
- preserve Task038 runner/model/action contract;
- support env4096 long runs;
- keep `num_envs % num_mini_batches == 0`;
- record sequence update counters and stateless fallback counters;
- write a checkpoint path and JSON summary;
- not claim clean-gait success without eval.

Required no-overclaim fields:

- `quality_claim:false`
- `training_claim:false`
- `eval_claim:false`
- `reproduction_claim:false`
- `superiority_claim:false`

## Log

- 2026-05-30 Added
  `src/h200_locomotion_lab/tools/task041_sequence_txl_clean_train.py`.
- 2026-05-30 Local focused tests passed after adding bridge-compatible actor
  config defaults:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp
  .agent/tmp/pytest-task041-quality-tol tests/test_task039_quality_feedback.py
  tests/test_task039_true_txl_clean_eval.py
  tests/test_task041_sequence_txl_clean_eval.py
  tests/test_task041_adaptk160_true_txl_warmstart.py
  tests/test_task041_sequence_txl_clean_train.py tests/test_agent_inventory.py`
  returned `43 passed, 1 skipped in 0.25s`.
- 2026-05-30 H200 smoke train had already verified the sequence-aware update
  path before the warmstart bridge:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/sequence_txl_clean_train/smoke_train_env8_iter1.json`
  with `train_pipeline_pass=true`, `algorithm_class=Task040SequenceAwareTrueTxlPPO`,
  `stateless_fallback_forward_batches=0`, and
  `sequence_update_forward_batches=1`.

## Review

Status: closed for train/update plumbing. Clean-gait quality is closed by the
Task041 eval gate, not by this train summary.
