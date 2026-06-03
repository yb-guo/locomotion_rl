# 011: Fault Aux Conditioning Probe

## Route

Subtask 010 proved that memory latent can influence actions, but the policy still
does not turn that influence into velocity recovery. The next diagnostic is a
Task044-only auxiliary fault-identification signal:

- keep fault identity out of actor-visible observations;
- expose the current hidden dynamic-fault target as a privileged training-only
  label group;
- add a PPO-side auxiliary classifier on memory-only latent, not on actor base
  observations;
- keep default behavior unchanged when the aux loss weight is zero;
- keep Task044 acceptance tied to normal/zero/stateless triplet eval, not aux
  loss alone.

## Acceptance

- Old checkpoints still load with `strict=True`.
- Local tests cover CLI/config propagation and aux train-pipeline gating.
- H200 smoke records `task044_fault_aux_updates > 0` and finite
  `task044_fault_aux` loss.
- H200 early-window aux smoke records that the trial filter is active. A
  one-iteration plumbing smoke may use `task044_fault_aux_min_trial_index=0`;
  the memory-required long run must use `task044_fault_aux_min_trial_index>=1`.
- H200 long train records train-pipeline evidence before any pass claim.
- H200 triplet eval decides whether the aux signal improves Task044; aux loss is
  not a memory-causality claim.

## Log

- 2026-05-31 Added `Task044FaultLabelVecEnvWrapper`, which adds
  `task044_fault_label` as a non-actor observation group. The Task044 actor and
  critic obs groups remain `actor_history` and `critic`.
- 2026-05-31 Added PPO-side fault auxiliary support to
  `Task040SequenceAwareTrueTxlPPO`. The auxiliary head is not part of the actor
  checkpoint, so old actor checkpoints can still resume with `strict=True`.
- 2026-05-31 The auxiliary classifier consumes memory-only sequence latents from
  `Task038TrueTxlMemoryModel`, not the full policy latent with base obs
  passthrough.
- 2026-05-31 Local validation passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task041_sequence_txl_clean_train.py::test_task041_sequence_txl_clean_train_parse_args_defaults tests\test_task041_sequence_txl_clean_train.py::test_task041_sequence_txl_clean_train_preflight_rejects_bad_values tests\test_task041_sequence_txl_clean_train.py::test_task041_sequence_txl_clean_train_mutates_cfg_to_sequence_aware_algorithm tests\test_task041_sequence_txl_clean_train.py::test_task041_train_pipeline_requires_fault_aux_updates_when_enabled tests\test_task044_hidden_fault_target.py --tb=short`
  with 9 passed and 1 skipped.
- 2026-05-31 H200 aux smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4401602.json`.
  Result: `train_pipeline_pass=true`, `task044_fault_aux_updates=20`, and
  last loss dict included `task044_fault_aux=1.7369526267051696`.
- 2026-05-31 Started H200 scale-4 aux continuation training from:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_clear_history_env1024_iter50_scale4_from_model49_seed4401401/model_49.pt`.
  Background PID: `589546`. Expected output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_scale4_aux002_env1024_iter50_from_scale4_model49_seed4401701.json`.
- 2026-05-31 H200 scale-4 aux continuation completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_scale4_aux002_env1024_iter50_from_scale4_model49_seed4401701.json`.
  The train pipeline passed with `task044_fault_aux_updates=1000`, but triplet
  eval failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/clear_history_scale4_aux002_iter50_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4401801.json`.
  Result: `task044_memory_required_pass=false`; normal quality failed and
  stateless-memory stayed tied.
- 2026-05-31 H200 bridge aux scale-1 continuation completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_bridge_aux002_scale1_env1024_iter50_seed4401901.json`.
  Eval failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/bridge_aux002_scale1_iter50_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4402001.json`.
  Stateless-memory action delta was only
  `mean_abs_l1_delta=0.0015200767065248182`, so whole-trial fault
  classification was not sufficient to force cross-trial cache use.
- 2026-05-31 Added early post-inner-reset aux filtering:
  `task044_trial_step`, `task044_trial_index`,
  `task044_fault_aux_max_trial_step`, and
  `task044_fault_aux_min_trial_index`. This targets the first steps after
  visible history is cleared, where stateless memory should not know the prior
  trial condition.
- 2026-05-31 Local validation for the early-window patch passed:
  `python -m h200_locomotion_lab.tools.task044_hidden_fault_train --help`,
  `python -m h200_locomotion_lab.tools.inspect_agent`, and targeted pytest
  with 9 passed and 1 skipped.
- 2026-05-31 H200 early-window plumbing smoke passed with
  `task044_fault_aux_max_trial_step=4` and
  `task044_fault_aux_min_trial_index=0`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4402102.json`.
  The same one-iteration smoke with `min_trial_index=1` correctly produced no
  aux updates because the rollout had not reached trial 1.
- 2026-05-31 H200 early-window bridge continuation completed with
  `task044_fault_aux_max_trial_step=4` and
  `task044_fault_aux_min_trial_index=1`:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_bridge_aux002_early4_trial1_scale1_env1024_iter50_seed4402201.json`.
  Train pipeline passed with `task044_fault_aux_updates=205`. Triplet eval
  failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/bridge_aux002_early4_trial1_scale1_iter50_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4402301.json`.
- 2026-05-31 H200 continued the early-window bridge run for another 50
  iterations:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_bridge_aux002_early4_trial1_scale1_env1024_iter50_cont2_seed4402401.json`.
  Train pipeline passed with `task044_fault_aux_updates=220` and
  `task044_fault_aux_last_loss=0.12627872824668884`. The action influence
  summary passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/action_influence_summary/bridge_aux002_early4_trial1_scale1_iter100_model49_actionstats_left_knee_joint_vx1p6_seed4402501.json`.
  The triplet still failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/bridge_aux002_early4_trial1_scale1_iter100_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4402501.json`.
  Result: `task044_memory_required_pass=false`; normal final-trial
  `lin_vel_error.mean=0.504334568977356`, zero-residual window delta
  `0.012218117713928223`, and stateless-memory window delta
  `0.011637210845947266`, all short of the triplet thresholds.

## Review

Status: failed diagnostic.

The aux training path is mechanically valid: strict resume works, the label
loss updates, early post-reset filtering works, and after 100 total iterations
the action-level influence summary passes. This is still not memory-required
locomotion evidence. The normal policy quality remains below the Task044 gate,
and the zero/stateless ablations do not degrade enough in the final-trial
window. The useful conclusion is narrower: fault labels can reach memory latent
and action statistics, but a label classifier alone does not create a strong
behavior-level adaptation objective for the current MLP+TXL residual policy.
