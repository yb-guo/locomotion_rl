# 008: Bridge Residual Trainable Scope

## Route

Subtask 007 proved that `model_5349` can be migrated into the True-TXL actor
and still pass the dynamic-switch quality gate. The first residual-only
training attempt from that bridge exposed a contract bug: the TXL branch had no
trainable path into the action head.

Fix the training contract before spending more H200 time:

- keep the copied MLP base-observation path frozen;
- keep the observation normalizer frozen for non-`all` scopes;
- train TXL residual parameters;
- additionally allow only the 32 memory-latent columns of `mlp.0.weight` to
  train;
- mask gradients on the frozen base-observation columns of `mlp.0.weight`;
- reject training summaries if those frozen columns drift.

## Acceptance

- Local tests prove the new scope masks base-observation gradients and only
  allows memory-latent columns to receive gradient.
- Local tests prove the train gate rejects frozen-column drift.
- H200 training with the new scope records nonzero TXL parameter deltas and
  `txl_residual_output_norm`.
- Eval evidence states whether the trained residual preserves quality and
  whether ablations degrade.

## Log

- 2026-05-31 H200 residual-only training from the bridge with the old
  `txl_residual_only` scope passed the train pipeline but changed no TXL
  parameters. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_model5349_bridge_residual_env1024_iter25_seed4301801.json`.
  The summary reported `memory_output_projection_delta_norm=0.0`,
  `attention_layers_delta_norm=0.0`, `token_projection_delta_norm=0.0`, and
  `txl_residual_output_norm_last=0.0`.
- 2026-05-31 Added train scope
  `txl_residual_and_mlp_memory_input` in
  `src/h200_locomotion_lab/tools/task041_sequence_txl_clean_train.py`. It
  trains the TXL residual parameters plus only the memory-latent columns of
  `mlp.0.weight`; a gradient hook masks the frozen base-observation columns.
- 2026-05-31 Local validation passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task041_sequence_txl_clean_train.py tests\test_task041_adaptk160_true_txl_warmstart.py tests\test_task043_dynamic_switch_training_contract.py --tb=short --basetemp pytest_tmp_task043_memory_input_scope`
  with sandbox escalation for pytest temp creation: 18 passed, 9 skipped.
- 2026-05-31 H200 1024-env, 25-iteration training with the new scope passed
  the train pipeline:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_model5349_bridge_residual_mlp_memory_input_env1024_iter25_seed4301901.json`.
  It produced:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/logs_model5349_bridge_residual_mlp_memory_input_env1024_iter25_seed4301901/model_24.pt`.
  Key train evidence: `task043_train_pipeline_pass=true`,
  `partial_trainable_delta_norm=1.3401391506195068`,
  `partial_frozen_delta_norm=0.0`,
  `memory_output_projection_delta_norm=0.9404060244560242`,
  `attention_layers_delta_norm=11.947895050048828`,
  `token_projection_delta_norm=4.212675094604492`,
  `txl_residual_output_norm_last=1.6352477073669434`.
- 2026-05-31 H200 model_24 normal eval completed without falling but failed the
  strict quality gate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_bridge_residual_mlp_memory_input_env1024_iter25_model24_switch_none_vx1p6_seed4302001.json`.
  Final trial: `lin_vel_error.mean=0.44369199872016907`, but
  `quality_gate_pass=false` due to
  `gravity_xy_max_regressed_from_trial0` and
  `root_z_min_regressed_from_trial0`.
- 2026-05-31 H200 model_24 zero-residual and stateless ablations did not show
  useful degradation:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_bridge_residual_mlp_memory_input_env1024_iter25_model24_switch_zero_residual_vx1p6_seed4302001.json`
  had final `lin_vel_error.mean=0.4421100914478302`; and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_bridge_residual_mlp_memory_input_env1024_iter25_model24_switch_stateless_vx1p6_seed4302001.json`
  had final `lin_vel_error.mean=0.44447004795074463`. Both remained
  behaviorally tied with normal mode.
- 2026-05-31 H200 shorter 5-iteration training also failed quality:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_switch_train/train_model5349_bridge_residual_mlp_memory_input_env1024_iter5_seed4302101.json`.
  Eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_bridge_residual_mlp_memory_input_env1024_iter5_model4_switch_none_vx1p6_seed4302201.json`.
  Final `lin_vel_error.mean=0.4562378227710724`,
  `quality_gate_pass=false`.

## Review

Status: passed for training-contract repair, failed for memory-required quality.

The original residual-only branch was invalid as a learning experiment because
the memory path had no trainable action connection. The new scope fixes that
mechanically and is covered by local tests plus H200 train evidence.

The resulting policy still does not satisfy Task043's memory-required goal. A
25-iteration run makes the residual active and keeps the robot upright, but the
strict quality gate fails on posture/height regression and ablations remain
tied. A short 5-iteration run also fails on velocity error. The current
dynamic-switch setup does not yet create a clean advantage for memory over the
high-speed MLP prior.
