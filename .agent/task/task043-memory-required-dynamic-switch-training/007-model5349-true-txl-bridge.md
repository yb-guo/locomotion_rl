# 007: Model5349 True-TXL Bridge

## Route

Fixed-speed Task043 training from the Task042 residual prior stayed just above
the quality gate. The next minimal diagnosis is whether the known high-speed
MLP prior, `model_5349`, can be converted into the True-TXL actor without
breaking the base gait.

This is a bridge check, not a memory-causality claim:

- copy the 104-dim MLP actor prefix into the True-TXL actor;
- zero the extra 32 memory-latent input columns in the first actor layer;
- expand the 104-dim observation normalizer across every K160 history frame;
- leave action-history normalizer dimensions at default mean 0 and std/var 1;
- evaluate normal, zero-residual, and stateless-memory modes on the same
  checkpoint, seed, speed, and dynamic switch.

## Acceptance

- Local tests cover partial first-layer copy and history normalizer expansion.
- H200 warmstart JSON reports `warmstart_pipeline_pass=true`.
- Normal Task043 dynamic-switch eval passes the selected quality gate.
- Ablation evals are recorded before any memory-causality conclusion.
- If ablations remain tied, the review explicitly states no positive memory
  causality.

## Log

- 2026-05-31 Initial `model_5349` bridge failed pipeline because actor keys had
  expected shape mismatches: `obs_normalizer._mean`, `obs_normalizer._var`,
  `obs_normalizer._std`, and `mlp.0.weight`.
- 2026-05-31 Added partial migration support in
  `src/h200_locomotion_lab/training/rsl_history_wrapper.py`: first-layer source
  columns are copied, memory-latent columns are zeroed, and the 104-dim source
  normalizer is expanded into every 135-dim history frame.
- 2026-05-31 Added/updated local tests in
  `tests/test_task041_adaptk160_true_txl_warmstart.py` for partial actor copy,
  partial-key warmstart acceptance, and history normalizer expansion.
- 2026-05-31 Local validation passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task041_adaptk160_true_txl_warmstart.py tests\test_task043_dynamic_switch_training_contract.py --tb=short --basetemp pytest_tmp_task043_model5349_history_norm`
  with sandbox escalation for pytest temp creation: 9 passed, 3 skipped.
- 2026-05-31 H200 warmstart passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/model5349_true_txl_warmstart/model_5349_task043_true_txl_bridge_history_norm_zero_tail.json`.
  It produced:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/model5349_true_txl_warmstart/model_5349_task043_true_txl_bridge_history_norm_zero_tail.pt`.
- 2026-05-31 H200 normal dynamic-switch eval passed the quality gate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_true_txl_bridge_history_norm_zero_tail_switch_none_vx1p6_seed4301701.json`.
  Final trial: `quality_gate_pass=true`, `fall_ratio=0.0`,
  `lin_vel_error.mean=0.4467167258262634`,
  `yaw_vel_error.mean=0.11126784980297089`, `root_z.min=0.7544786334037781`.
- 2026-05-31 H200 zero-residual ablation remained tied:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_true_txl_bridge_history_norm_zero_tail_switch_zero_residual_vx1p6_seed4301701.json`.
  Final trial: `quality_gate_pass=true`,
  `lin_vel_error.mean=0.44687220454216003`,
  `yaw_vel_error.mean=0.10972824692726135`.
- 2026-05-31 H200 stateless-memory ablation also remained tied:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task043/dynamic_ablation_eval/model5349_true_txl_bridge_history_norm_zero_tail_switch_stateless_vx1p6_seed4301701.json`.
  Final trial: `quality_gate_pass=true`,
  `lin_vel_error.mean=0.446464478969574`,
  `yaw_vel_error.mean=0.11053020507097244`. The top-level
  `pipeline_pass=false` is expected for this destructive ablation because the
  stateless mode intentionally disables incremental memory exposure; the
  `quality_feedback.pipeline_pass=true` field confirms the quality metrics were
  still computed.

## Review

Status: passed for high-speed prior bridge quality, failed for positive memory
causality.

The migration bug was real: copying only the first normalizer prefix made the
bridge fall, while expanding the base normalizer across all history frames
restored the high-speed gait. This closes the base-prior blocker for Task043.

It does not close the memory-required objective. The bridge checkpoint has
`txl_residual_output_norm=0.0`, and normal, zero-residual, and stateless-memory
evals are behaviorally tied. The next useful experiment is residual-only
training from this bridge checkpoint, because the base gait now passes the
dynamic-switch quality gate and any later ablation degradation would be easier
to attribute to the trained memory residual.
