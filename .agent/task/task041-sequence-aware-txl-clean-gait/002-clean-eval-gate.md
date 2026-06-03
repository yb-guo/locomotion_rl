# 002: Clean Eval Gate

## Route

Task041 reuses the Task039 quality gate. The only acceptable close condition is
clean eval pass on the sequence-aware checkpoint.

## Contract

Eval pass requires:

- `task041_sequence_txl_clean_eval=true`;
- `pipeline_pass=true`;
- `quality_gate_pass=true`;
- `pass=true`;
- clean command `lin_vel_x=0.4`;
- active true-TXL inference memory debug from the eval rollout;
- no-overclaim flags false.

The eval wrapper may optionally take a train summary JSON. If provided, it must
match the checkpoint and show `train_pipeline_pass=true`.

Required no-overclaim fields:

- `quality_claim:false`
- `training_claim:false`
- `eval_claim:false`
- `reproduction_claim:false`
- `superiority_claim:false`

## Log

- 2026-05-30 Added
  `src/h200_locomotion_lab/tools/task041_sequence_txl_clean_eval.py`.
- 2026-05-30 Fixed the eval memory-debug boundary by carrying
  `txl_debug` from `task037_multitrial_eval_checkpoint.py` into the Task039
  wrapper. H200 smoke eval then changed from `memory_debug_missing` to
  `pipeline_pass=true` with quality-only failure.
- 2026-05-30 H200 warmstart bridge clean eval passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task041/sequence_txl_clean_eval/model_5467_task041_true_txl_bridge_vx0p4_eval_tolerance1e3.json`
  with `task041_sequence_txl_clean_eval=true`, `pipeline_pass=true`,
  `task041_pipeline_pass=true`, `quality_gate_pass=true`, `pass=true`, and
  `memory_debug_active=true`.

## Review

Status: passed for clean 0.4 m/s eval gate. This remains diagnostic-only and
does not claim reproduction or superiority.
