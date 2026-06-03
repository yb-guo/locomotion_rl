# 013: Velocity Pressure Left-Knee Stage

## Route

Subtask 012 aligned the train schedule to the triplet eval and produced stable
left-knee-dead locomotion, but the policy converged to a slow gait. The strict
quality gate still failed on linear velocity error. This subtask keeps the eval
gate unchanged and adds one training-only pressure stage:

- same hidden left-knee dead schedule as subtask 012;
- same 2.0 s trial length and fixed `vx=1.6`;
- same runner, actor, observation, and action contract;
- increase only the training `track_linear_velocity` reward pressure.

This is a targeted training-stage diagnostic. Passing requires the normal
triplet quality gate and behavior ablation degradation, not train reward alone.

## Acceptance

- Local tests lock the velocity-pressure task id and helper.
- H200 registry contains the new velocity-pressure task id.
- H200 smoke passes for the new task.
- H200 continuation train records train-pipeline evidence.
- H200 triplet eval decides pass/fail with the original Task044 thresholds.

## Log

- 2026-05-31 Added
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-EvalLeftKneeVelBoost1p6`.
  It wraps the eval-aligned left-knee helper and sets
  `track_linear_velocity.weight=3.0` plus `track_linear_velocity.std=1.0`.
- 2026-05-31 Local validation passed:
  `python -m pytest -q -p no:cacheprovider tests\test_task044_hidden_fault_target.py`
  with 5 passed and 1 skipped.
- 2026-05-31 H200 registry patch applied. Registry contains the velocity-pressure
  task id.
- 2026-05-31 H200 smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4403201.json`.
- 2026-05-31 Started H200 velocity-pressure continuation from:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_evalaligned_leftknee_aux002_early4_trial1_scale1_env1024_iter100_cont_seed4403001/model_99.pt`.
  Background PID: `601174`. Expected output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_evalaligned_leftknee_velboost_aux002_early4_trial1_scale1_env1024_iter50_seed4403301.json`.
- 2026-05-31 H200 velocity-pressure continuation completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_evalaligned_leftknee_velboost_aux002_early4_trial1_scale1_env1024_iter50_seed4403301.json`.
  Result: `train_pipeline_pass=true`, `task044_fault_aux_updates=830`.
- 2026-05-31 H200 triplet eval failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/evalaligned_leftknee_velboost_aux002_early4_trial1_scale1_iter50_model49_actionstats_triplet_left_knee_joint_vx1p6_seed4403401.json`.
  Final trial remained stable, and speed improved to
  `lin_vel_error.mean=0.8055592775344849`, but this is still above the strict
  gate. Window ablation deltas stayed too small.
- 2026-05-31 Eval-only `memory_latent_scale=2.0` probe failed by destabilizing
  normal mode:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/evalaligned_leftknee_velboost_aux002_early4_trial1_scale1_iter50_model49_evalscale2_actionstats_triplet_left_knee_joint_vx1p6_seed4403501.json`.
- 2026-05-31 Started another velocity-pressure continuation from the
  velocity-pressure `model_49` checkpoint. Background PID: `603653`. Expected
  output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_evalaligned_leftknee_velboost_aux002_early4_trial1_scale1_env1024_iter100_cont_seed4403601.json`.
- 2026-05-31 H200 longer velocity-pressure continuation completed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_evalaligned_leftknee_velboost_aux002_early4_trial1_scale1_env1024_iter100_cont_seed4403601.json`.
  Result: `train_pipeline_pass=true`, checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_evalaligned_leftknee_velboost_aux002_early4_trial1_scale1_env1024_iter100_cont_seed4403601/model_99.pt`.
- 2026-05-31 H200 triplet eval for the longer continuation failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/evalaligned_leftknee_velboost_aux002_early4_trial1_scale1_iter150_model99_actionstats_triplet_left_knee_joint_vx1p6_seed4403701.json`.
  Final trial stayed stable but still too slow:
  `lin_vel_error.mean=0.7220476269721985`; zero-residual and stateless
  ablations did not materially degrade.
  A 1.0 s window diagnostic:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/evalaligned_leftknee_velboost_aux002_early4_trial1_scale1_iter150_model99_window1_actionstats_triplet_left_knee_joint_vx1p6_seed4403801.json`
  showed zero-residual degradation but stateless-memory remained tied.

## Review

Status: closed as a failed diagnostic.

Velocity-pressure improved speed but did not reach the strict quality gate and
did not create a stateless-memory behavior gap. This subtask is not a reward
contract change for eval; it remains failed evidence for the observed slow
stable gait.
