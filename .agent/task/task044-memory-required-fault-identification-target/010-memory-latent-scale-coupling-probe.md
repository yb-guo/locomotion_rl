# 010: Memory Latent Scale Coupling Probe

## Route

Subtask 009 showed that clearing visible history is not enough: the old and
newly trained checkpoints still keep stateless-memory behavior close to normal.
The next hypothesis is weak policy coupling: memory latent is nonzero, but the
MLP output is not sensitive enough to it.

This subtask adds a scalar coupling knob:

- `memory_latent_scale` multiplies the memory latent before it is concatenated
  with newest base observation;
- default is `1.0`, so existing checkpoints and behavior remain unchanged;
- no checkpoint parameter shape changes are introduced;
- the knob is available to eval and train CLIs.

## Acceptance

- Local tests cover parse/config propagation.
- H200 eval-only scale probe records whether action influence increases.
- H200 scale train records pipeline evidence before any success claim.
- Task044 still requires normal/zero/stateless triplet evidence; scale action
  deltas alone are not a pass.

## Log

- 2026-05-31 Added `memory_latent_scale` to `Task038TrueTxlMemoryModel`.
  The scale is applied after adaptation + TXL residual composition and before
  the MLP input.
- 2026-05-31 Added `--memory-latent-scale` to Task041 eval/train CLIs and to
  Task037 eval config forwarding.
- 2026-05-31 Local validation passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task041_sequence_txl_clean_eval.py::test_task041_sequence_txl_clean_eval_parse_args_defaults tests\test_task041_sequence_txl_clean_train.py::test_task041_sequence_txl_clean_train_parse_args_defaults tests\test_task041_sequence_txl_clean_train.py::test_task041_sequence_txl_clean_train_preflight_rejects_bad_values tests\test_task041_sequence_txl_clean_train.py::test_task041_sequence_txl_clean_train_mutates_cfg_to_sequence_aware_algorithm tests\test_task037_multitrial_contract.py tests\test_task044_hidden_fault_target.py tests\test_task037_mjlab_smoke_scripts.py tests\test_task044_action_influence_contract.py tests\test_task044_action_influence_summary_cli.py --tb=short`
  with 24 passed and 7 skipped.
- 2026-05-31 H200 eval-only `memory_latent_scale=4.0` probe on
  clear-history model_49 recorded:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/action_influence_summary/clear_history_env1024_iter50_model49_scale4_actionstats_left_knee_joint_vx1p6_seed4401301.json`.
  Result: action influence became detectable with no action-summary failure
  reasons. Zero-residual action delta was
  `mean_abs_l1_delta=0.06077523926092732`; stateless-memory action delta was
  `mean_abs_l1_delta=0.014938026426299926`.
- 2026-05-31 The scale-4 eval-only triplet still failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/clear_history_env1024_iter50_model49_scale4_actionstats_triplet_left_knee_joint_vx1p6_seed4401301.json`.
  Failure reasons were `normal_quality_gate_not_passed` and
  `stateless_memory_ablation_not_degraded`.
- 2026-05-31 Started H200 scale-4 continuation training from
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_clear_history_env1024_iter50_seed4401101/model_49.pt`.
  Background PID: `587141`.
- 2026-05-31 H200 scale-4 continuation training completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_env1024_iter50_scale4_from_model49_seed4401401.json`.
  Result: `train_pipeline_pass=true`, checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_clear_history_env1024_iter50_scale4_from_model49_seed4401401/model_49.pt`.
- 2026-05-31 H200 scale-4 trained checkpoint triplet eval failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/clear_history_scale4_iter50_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4401501.json`.
  Result: `task044_memory_required_pass=false` with
  `normal_quality_gate_not_passed`, `zero_residual_ablation_not_degraded`, and
  `stateless_memory_ablation_not_degraded`.
- 2026-05-31 H200 scale-4 trained checkpoint action influence was detectable:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/action_influence_summary/clear_history_scale4_iter50_model49_actionstats_left_knee_joint_vx1p6_seed4401501.json`.
  Zero-residual action delta was
  `mean_abs_l1_delta=0.11061656042452782`; stateless-memory action delta was
  `mean_abs_l1_delta=0.042740989836954305`. This is diagnostic evidence only.
- 2026-05-31 The scale-4 trained normal eval still failed locomotion quality.
  In the final-trial 0.5s window, normal `lin_vel_error.mean` was
  `1.7986218929290771`, `fall_ratio=0.0`, `zero_fall_ratio=1.0`,
  `gravity_xy.max=0.2589489817619324`, and `root_z.min=0.7099651098251343`.
  The checkpoint avoided falling in that window but did not track the 1.6 m/s
  command.

## Review

Status: closed as a failed diagnostic.

The coupling hypothesis is partially supported and bounded. Scaling the latent
creates action-level differences, so memory can reach the policy output. It does
not solve the rollout target: after scale-4 continuation training, normal mode
still fails quality and both ablations fail to degrade enough by Task044
criteria. The next route should not be another blind latent-scale run. It should
add a stronger memory-specific training signal or policy consumer that makes
hidden-fault identification useful for velocity recovery.
