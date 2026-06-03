# 009: Clear Visible History Inner-Reset Runner

## Route

Subtask 008 showed that stateful and stateless TXL actions are effectively tied.
One possible confound is the Task037 contract: inner trial reset preserves
actor-visible K160 history. That means stateless-memory mode can still see
recent pre-reset history through `actor_history`, even when the TXL cache is
disabled.

This subtask separates the two memory channels:

- clear actor-visible history on inner trial reset;
- preserve TXL cache across inner trial reset;
- keep outer episode reset clearing both;
- keep this behavior scoped to Task044 so earlier Task037 evidence remains
valid.

## Acceptance

- Local tests cover the history wrapper's new `clear_history_on_inner_reset`
  mode.
- Task044 registration uses a Task044-specific clear-history runner.
- Task044 eval expects the clear-history runner and still records no-overclaim
  fields.
- H200 registration is patched and verified.
- H200 action diagnostic is rerun on the existing checkpoint before retraining.

## Log

- 2026-05-31 Added `clear_history_on_inner_reset` to
  `Task033HistoryVecEnvWrapper`; default remains `False`.
- 2026-05-31 Added `Task044TrueTxlMemoryK160ClearHistoryRunner`, which clears
  visible `actor_history` on inner reset and preserves TXL cache through
  `Task038TrueTxlResetHookVecEnvWrapper`.
- 2026-05-31 Updated
  `.agent/task/task044-memory-required-fault-identification-target/task044_register_hidden_fault_stage.py`
  so the Task044 H200 MJLab registration uses the clear-history runner.
- 2026-05-31 Updated Task044 eval expected runner to
  `Task044TrueTxlMemoryK160ClearHistoryRunner`.
- 2026-05-31 Local validation passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task037_multitrial_contract.py tests\test_task044_hidden_fault_target.py tests\test_task037_mjlab_smoke_scripts.py tests\test_task044_action_influence_contract.py tests\test_task044_action_influence_summary_cli.py --tb=short`
  with 20 passed and 7 skipped.
- 2026-05-31 H200 registration patched:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/src/tasks/velocity/config/g1_gripper/__init__.py`
  now imports `Task044TrueTxlMemoryK160ClearHistoryRunner` and registers it for
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-Fast1p6`.
- 2026-05-31 H200 clear-history action diagnostic on the existing `model_24`
  checkpoint recorded:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/action_influence_summary/model5349_hidden_fault_env1024_iter25_model24_actionstats_left_knee_joint_vx1p6_seed4400901.json`.
  The eval JSONs report `runner_cls=Task044TrueTxlMemoryK160ClearHistoryRunner`.
  Result: `stateless_memory_action_stats_tied` remains. Stateless action
  deltas were `mean_abs_l1_delta=0.0016770426544450944` and
  `mean_l2_delta=-0.004179716110229492`.
- 2026-05-31 H200 clear-history train smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4401001.json`.
  Result: `train_pipeline_pass=true`,
  `task044_train_pipeline_pass=true`,
  `runner_cls=Task044TrueTxlMemoryK160ClearHistoryRunner`,
  `algorithm_class=Task040SequenceAwareTrueTxlPPO`, checkpoint
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_clear_history_smoke_env64_iter1_seed4401001/model_0.pt`.

## Review

Status: implementation and smoke closed; long retraining pending.

The visible-history confound is now controlled for Task044. The old checkpoint
still does not use stateful TXL memory after that control, so the next required
step is not more eval on the same checkpoint. Long-train under the clear-history
runner and repeat the normal/zero/stateless triplet plus action diagnostic.
