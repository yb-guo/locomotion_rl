# 005: Residual-Only Train Guardrail

## Route

Subtask 003 showed that unrestricted PPO continuation makes the TXL residual
non-zero but breaks the clean gait prior. The next smallest fix is to keep the
Task041 AdaptK160 bridge as a frozen baseline and restrict actor updates to the
TXL residual path.

Add an explicit train-time `actor_trainable_scope`:

- `all`: existing Task041 behavior;
- `txl_residual_only`: train token projection, TXL attention/norm, position
  embedding, and memory output projection;
- `memory_output_projection_only`: most conservative smoke, training only the
  residual projection.

The first H200 probe should use `memory_output_projection_only` from the
Task041 bridge checkpoint. If clean gait survives and residual output changes,
then widen to `txl_residual_only`.

## Acceptance

- Local tests cover trainable-scope parsing, freezing, and frozen-delta gates.
- Train summary records:
  - `actor_trainable_scope`;
  - trainable/frozen parameter names;
  - whether actor normalization updates were disabled;
  - trainable parameter delta;
  - frozen parameter delta;
  - frozen obs-normalizer delta.
- Pipeline pass rejects non-`all` scope if frozen parameters or normalizer
  buffers change.
- H200 summary proves the selected scope has no frozen drift and at least one
  residual-path delta is non-zero.
- Clean 0.4 m/s eval is rerun on the produced checkpoint before any memory
  causality claim.

## Log

- 2026-05-30 Opened after subtask 003 failed on unrestricted residual
  continuation. Implementation starts with the existing Task041 train wrapper
  rather than a new runner class.
- 2026-05-30 Implemented `--actor-trainable-scope` in the Task041 train
  wrapper with modes `all`, `txl_residual_only`, and
  `memory_output_projection_only`. For non-`all` scopes the wrapper freezes
  disallowed actor parameters and disables actor normalization updates.
- 2026-05-30 Local verification:
  - `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests/test_task041_sequence_txl_clean_train.py tests/test_task042_memory_ablation_contract.py --basetemp (Join-Path $env:TEMP 'task042_scope_pytest')`
    -> `13 passed, 4 skipped`;
  - `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task041_sequence_txl_clean_train --help`
    shows `--actor-trainable-scope`;
  - `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.inspect_agent`
    passed.
- 2026-05-30 H200 `memory_output_projection_only` probe completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/residual_only_train/projection_only_env1024_iter5_summary.json`.
  Result: `train_pipeline_pass=true`, `failure_reasons=[]`,
  `trainable_parameter_count=4128`, `frozen_parameter_count=3159518`,
  `normalization_update_disabled=true`, `frozen_parameter_delta_norm=0.0`,
  `frozen_obs_normalizer_delta_norm=0.0`,
  `memory_output_projection_delta_norm=1.0988479852676392`,
  `txl_residual_output_norm_last=1.5966463088989258`,
  `sequence_update_forward_batches=20`, and
  `stateless_fallback_forward_batches=0`.
- 2026-05-30 H200 `txl_residual_only` probe completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task042/residual_only_train/txl_residual_only_env1024_iter5_summary.json`.
  Result: `train_pipeline_pass=true`, `failure_reasons=[]`,
  `trainable_parameter_count=156192`, `frozen_parameter_count=3007454`,
  `normalization_update_disabled=true`, `frozen_parameter_delta_norm=0.0`,
  `frozen_obs_normalizer_delta_norm=0.0`,
  `memory_output_projection_delta_norm=0.7584468126296997`,
  `attention_layers_delta_norm=9.425192832946777`,
  `norm_layers_delta_norm=0.8757206797599792`,
  `token_projection_delta_norm=3.1532840728759766`,
  `position_embedding_delta_norm=0.8367196917533875`,
  `txl_residual_output_norm_last=0.9989228248596191`,
  `sequence_update_forward_batches=20`, and
  `stateless_fallback_forward_batches=0`.

## Review

Status: passed for guardrail implementation and bounded H200 train evidence.

The guardrail proves the previous baseline-drift failure mode is controllable:
only the projection parameters changed, while the frozen actor path and actor
normalizer buffers stayed unchanged. The wider `txl_residual_only` scope also
kept frozen state unchanged while updating all intended residual groups. This
does not by itself prove memory causality; that remains dependent on subtask
003 eval deltas.
