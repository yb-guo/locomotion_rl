# 001: Memory Ablation Contract

## Route

Add the smallest test-covered contract that lets eval distinguish normal
true-TXL inference from ablated memory inference.

The first modes are:

- `none`: current Task041 behavior.
- `zero_txl_residual`: keep the base observation and AdaptK warmstart path, but
  force the learned TXL residual contribution to zero.
- `stateless_txl_memory`: keep the actor shape but prevent cross-step memory
  from being reused, so the eval reports whether the policy depends on carried
  memory rather than only the current history vector.

## Acceptance

- A local unit test can build/parse each ablation mode.
- Eval summaries include `memory_ablation_mode`.
- Eval summaries include whether the residual was enabled.
- Eval summaries preserve existing Task041 pass/no-overclaim behavior.
- Unsupported modes fail loudly before H200 runtime.

## Log

- 2026-05-30 Opened. No implementation yet.
- 2026-05-30 Implemented the first ablation contract:
  - `Task038TrueTxlMemoryModel.task042_set_memory_ablation_mode`;
  - `zero_txl_residual` zeros only the TXL residual contribution and keeps the
    AdaptK warmstart path;
  - `stateless_txl_memory` disables carried inference cache while preserving
    debug fields;
  - `task042_memory_ablation_eval.py` wraps Task041 eval and records
    no-overclaim memory ablation summaries.
- 2026-05-30 Local verification:
  - `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task042_memory_ablation_eval --help`;
  - `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp (Join-Path $env:TEMP 'pytest-task042-ablation-passfix') tests/test_task042_memory_ablation_contract.py tests/test_task041_sequence_txl_clean_eval.py tests/test_agent_inventory.py`;
  - result: `12 passed in 0.11s`.
- 2026-05-30 H200 ablation eval on the Task041 bridge checkpoint:
  - normal:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/model5467_bridge_none_vx0p4_seed4100201.json`;
    `pass=true`, `quality_gate_pass=true`, `task042_pass=true`,
    `memory_residual_enabled=true`, `stateful_memory_enabled=true`,
    `txl_residual_raw_norm=0.0`, `txl_residual_output_norm=0.0`;
  - zero residual:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/model5467_bridge_zero_txl_residual_vx0p4_seed4100201.json`;
    `pass=true`, `quality_gate_pass=true`, `task042_pass=true`,
    `memory_residual_enabled=false`, `stateful_memory_enabled=true`,
    `txl_residual_raw_norm=0.0`, `txl_residual_output_norm=0.0`;
  - stateless memory:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/memory_ablation_eval/model5467_bridge_stateless_txl_memory_vx0p4_seed4100201.json`;
    `quality_gate_pass=true`, `task042_pass=true`,
    `task041_pipeline_pass=false`, `memory_debug_active=false`,
    `stateful_memory_enabled=false`,
    `stateless_fallback_forward_batches=360`.
  Interpretation: the ablation controls work. The current bridge checkpoint
  has zero TXL residual contribution, so it is not memory-causality evidence.

## Review

Status: passed for ablation-contract plumbing. The current checkpoint remains
an AdaptK160 warmstart baseline with active TXL plumbing; it does not yet prove
TXL memory influence.
