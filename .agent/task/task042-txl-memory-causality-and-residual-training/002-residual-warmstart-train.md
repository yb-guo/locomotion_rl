# 002: Residual Warmstart Train

## Route

Continue from the Task041 bridge checkpoint and let Task040 sequence-aware PPO
update the true-TXL residual path.

The first pass is a bounded H200 run. It is not a quality claim by itself; it
must produce a checkpoint and measurable parameter-delta evidence.

## Acceptance

- H200 train summary has `train_pipeline_pass=true`.
- Sequence update counters are active.
- `stateless_fallback_forward_batches=0`.
- Summary records TXL residual parameter deltas, at minimum for
  `memory_output_projection`.
- Produced checkpoint exists and is used by subtask 003 eval.

## Log

- 2026-05-30 Opened. Source warmstart checkpoint is
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/adaptk160_true_txl_warmstart/model_5467_task041_true_txl_bridge.pt`.
- 2026-05-30 Added train-summary TXL parameter delta stats to
  `task041_sequence_txl_clean_train.py`. Local verification:
  - `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task041_sequence_txl_clean_train --help`;
  - `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp (Join-Path $env:TEMP 'pytest-task042-train-stats') tests/test_task041_sequence_txl_clean_train.py tests/test_task042_memory_ablation_contract.py tests/test_agent_inventory.py`;
  - result: `14 passed, 1 skipped in 0.14s`.
- 2026-05-30 H200 bounded residual train passed:
  - summary:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/residual_warmstart_train/env1024_iter5_summary.json`;
  - checkpoint:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/residual_warmstart_train/logs_env1024_iter5/model_4.pt`;
  - `train_pipeline_pass=true`;
  - `sequence_update_batches=20`;
  - `sequence_update_forward_batches=20`;
  - `stateless_fallback_forward_batches=0`;
  - `txl_residual_output_norm_last=0.377013623714447`;
  - `memory_output_projection_delta_norm=0.1979231983423233`;
  - `attention_layers_delta_norm=0.3016187846660614`;
  - `token_projection_delta_norm=0.11119281500577927`.

## Review

Status: passed for bounded residual-training evidence. The TXL residual path
left zero, but quality and memory-causality must be checked by subtask 003.
