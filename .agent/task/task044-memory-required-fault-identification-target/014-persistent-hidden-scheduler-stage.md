# 014: Persistent Hidden Scheduler Stage

## Route

Subtasks 012 and 013 showed that an eval-aligned fixed left-knee stage can
produce stable locomotion, but it also removes the memory requirement:
stateless-memory ablation can learn the same fixed left-knee gait. The next
hypothesis is that the hidden fault must stay fixed across inner trial resets
while remaining randomized across outer episodes.

This stage adds one MJLab scheduler option and one Task044 training task:

- `preserve_schedule_across_inner_resets=False` by default for existing tasks;
- `PersistentHiddenVelBoost1p6` sets it to `True`;
- the new task keeps `vx=1.6`, 2.0 s trials, velocity-pressure reward, and the
  Task044 clear-visible-history runner;
- the actor still receives no direct fault identity, severity, onset, or
  recovery labels.

The intended check is whether normal mode can use cross-trial memory to recover
while stateless-memory degrades on the same hidden schedule.

## Acceptance

- Local tests lock the scheduler patch, default `False`, persistent task
  registration, train allowlist, eval allowlist, and non-actor fault metadata.
- H200 scheduler patch applies to the external MJLab env config.
- H200 registry contains the persistent-hidden task id.
- H200 smoke passes with the persistent-hidden task id.
- H200 continuation train records train-pipeline evidence.
- H200 triplet eval decides pass/fail with the original Task044 behavior gate.

## Log

- 2026-05-31 Added
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenVelBoost1p6`.
  The helper fixes `vx=1.6`, sets `episode_length_s=2.0`, increases velocity
  reward pressure, biases toward left-knee dead dynamic cases, and sets
  `preserve_schedule_across_inner_resets=True`.
- 2026-05-31 Added
  `task044_patch_dynamic_training_scheduler.py` to patch the H200 MJLab
  dynamic-training scheduler. Existing tasks keep the default
  `preserve_schedule_across_inner_resets=False`.
- 2026-05-31 Local validation passed:
  `python -m pytest -q -p no:cacheprovider tests\test_task044_hidden_fault_target.py tests\test_task044_memory_required_contract.py tests\test_task044_triplet_summary_cli.py tests\test_task044_action_influence_contract.py tests\test_task044_action_influence_summary_cli.py tests\test_task037_mjlab_smoke_scripts.py tests\test_agent_inventory.py --tb=short --basetemp .test_tmp_task044_persistent_hidden_local`
  with 34 passed and 2 skipped.
- 2026-05-31 Local inventory passed:
  `python -m h200_locomotion_lab.tools.inspect_agent`.
- 2026-05-31 H200 scheduler and registry patches applied under
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab`.
  Verified `preserve_schedule_across_inner_resets` in `env_cfgs.py` and the
  persistent-hidden task registration in `g1_gripper/__init__.py`.
- 2026-05-31 H200 persistent-hidden smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4403901.json`.
- 2026-05-31 H200 persistent-hidden 50-iteration continuation completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_persistent_hidden_velboost_aux002_early4_trial1_scale1_env1024_iter50_seed4404001.json`.
  Checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_persistent_hidden_velboost_aux002_early4_trial1_scale1_env1024_iter50_seed4404001/model_49.pt`.
  Triplet eval failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/persistent_hidden_velboost_aux002_early4_trial1_scale1_iter50_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4404101.json`.
  Full-final speed improved to `lin_vel_error.mean=0.5294605493545532`, but
  rare instability failed the normal quality gate and 0.5 s window ablations
  remained tied.
- 2026-05-31 H200 persistent-hidden continuation to 100 total iterations
  completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_persistent_hidden_velboost_aux002_early4_trial1_scale1_env1024_iter100_cont_seed4404201.json`.
  Checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_persistent_hidden_velboost_aux002_early4_trial1_scale1_env1024_iter100_cont_seed4404201/model_49.pt`.
  Triplet eval failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/persistent_hidden_velboost_aux002_early4_trial1_scale1_iter100_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4404301.json`.
  Full-final speed remained near the gate (`lin_vel_error.mean=0.5500046610832214`),
  but the first 0.5 s window stayed slow and ablations stayed tied.
- 2026-05-31 Diagnosed a training/eval timing mismatch in the external dynamic
  training scheduler: dynamic-single training sampled onset from 1.0 to 4.0 s,
  while Task044 eval applies the dead motor from 0.0 s. Added scheduler support
  for `dynamic_single_onset_range_s` and `dynamic_single_duration_range_s` with
  defaults preserving existing behavior, plus the new task
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateVelBoost1p6`.
- 2026-05-31 H200 immediate-onset smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4404501.json`.
- 2026-05-31 H200 immediate-onset continuation from the persistent-hidden
  checkpoint was aborted after early training showed severe instability:
  `fell_over=1.2083` and iteration time increased to about 48 s. Partial stdout:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/persistent_immediate_velboost_aux002_early4_trial1_scale1_env1024_iter50_seed4404601.stdout.log`.
  No pass claim was made.

## Review

Status: open.

This subtask is a target-design repair, not a pass claim. Passing requires a
triplet summary with `task044_memory_required_pass=true`; smoke, train success,
or action influence only count as pipeline/diagnostic evidence.

Current diagnosis: delayed-onset randomized training can improve full-final
speed but does not train the first 0.5 s eval window. Immediate-onset randomized
training matches the eval timing but is too hard from the current checkpoint and
destabilizes early. The next repair should be a curriculum, not another blind
long run: first recover stable immediate-onset gait under a narrower target,
then reintroduce randomized hidden-fault identity.
