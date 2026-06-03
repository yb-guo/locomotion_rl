# 001 History Buffer Contract

## Route

Define the shared history stream before implementing any policy-specific
consumer.

Actor-visible frame fields:

- current actor observation;
- previous action already visible in the current actor contract;
- optional action-response residuals derived only from actor-visible state.

Debug-only fields:

- active failure joint id;
- motor scale;
- failure type;
- segment/case id;
- scheduler state.

Device and shape constraints:

- Buffer must live on GPU for H200 training/eval.
- No Python per-env loops in step-time append/reset.
- Ring buffer layout should be batched:
  `[num_envs, history_len, frame_dim]` or equivalent.
- Reset must clear or reinitialize only the reset `env_ids`.
- Initial Task033 actor frame is `actor_obs[104] + prev_action[31]`
  unless a later H200 adapter patch explicitly adds actor-visible residuals.
- StackMLP `K=4` input dim is therefore `4 * (104 + 31) = 540`.

Policy boundaries:

- StackMLP may flatten `K` frames.
- GRU may use current frame plus hidden state, but reset semantics still come
  from the shared buffer/done stream.
- LocoFormer-style consumer may tokenize history, but it must not own a
  separate env history path.

Evidence schema:

- `evidence_json_path`
- `command`
- `repo_commit`
- `h200_checkout`
- `host`
- `gpu_name`
- `device`
- `num_envs`
- `history_len`
- `base_actor_obs_dim`
- `action_dim`
- `actor_frame_dim`
- `actor_input_dim_stack`
- `storage_shape`
- `latest_shape`
- `flat_shape`
- `dtype`
- `buffer_device`
- `reset_env_ids`
- `reset_policy`
- `valid_counts`
- `reset_env_zero_prefix`
- `actor_field_count`
- `debug_field_count`
- `actor_fault_leakage_check`
- `pass`
- `failure_reasons`

## Log

- 2026-05-28 Planned as the first Task033 implementation contract.
- 2026-05-28 Added `TorchHistoryBuffer` in
  `src/h200_locomotion_lab/training/history_buffer.py` and smoke CLI
  `python -m h200_locomotion_lab.tools.task033_history_buffer_smoke`.
  The buffer is batched, reset-aware, lazy-imports torch, and rejects actor
  field names containing fault/failure/motor-scale labels.
- 2026-05-28 Local validation passed with:
  `PYTHONPATH=src python -m h200_locomotion_lab.tools.inspect_agent`;
  `PYTHONPATH=src python -m h200_locomotion_lab.tools.task033_history_buffer_smoke --help`;
  `PYTHONPATH=src python -m pytest -p no:cacheprovider tests/test_agent_inventory.py tests/test_task033_history_buffer.py`
  (`4 passed, 4 skipped`; torch-dependent tests skipped locally because torch
  is not installed).
- 2026-05-28 H200 CUDA smoke passed with 8192 envs, `history_len=4`,
  `actor_frame_dim=135`, StackMLP input dim `540`, storage
  `[8192, 4, 135]`, `buffer_device=cuda:0`, `gpu_name=NVIDIA H20D`,
  `is_gpu_resident=true`, `actor_fault_leakage_check=passed`,
  `reset_env_valid_count=2`, `reset_env_zero_prefix=2`, and
  `failure_reasons=[]`.
  Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task033/history_buffer_smoke/task033_history_buffer_cuda_smoke.json`.

## Review

Status: passed for the shared buffer contract smoke. This does not validate
StackMLP/GRU/LocoFormer consumers yet.
