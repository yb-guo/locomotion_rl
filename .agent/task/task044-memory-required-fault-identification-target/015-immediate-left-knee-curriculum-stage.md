# 015: Immediate Left-Knee Curriculum Stage

## Route

Subtask 014 found the concrete timing mismatch: delayed dynamic-single training
uses onset `1.0-4.0 s`, while the Task044 eval gate applies the dead motor from
`0.0 s`. The first immediate randomized continuation was too hard from the
current checkpoint and destabilized early.

This stage adds the narrowest immediate-onset curriculum slice before returning
to randomized hidden identity:

- fixed `vx=1.6`;
- 2.0 s trials;
- dynamic-single onset fixed to `0.0 s`;
- duration fixed to `2.0 s`;
- target fixed to `left_knee_joint`;
- failure fixed to dead motor with scale `0.0`;
- preserve the hidden schedule across inner resets;
- keep actor-visible observations unchanged.

This is a curriculum bridge. It can close only if it improves immediate-onset
stability and records eval evidence; it is not a memory-required pass unless
the triplet summary passes the full Task044 contract.

## Acceptance

- Local tests lock the new task id, registry helper, exact left-knee/dead
  settings, train allowlist, and eval allowlist.
- H200 registry contains the new task id.
- H200 smoke passes.
- H200 continuation train records train-pipeline evidence.
- H200 eval records whether immediate-onset left-knee stability improved.
- Review explicitly states whether this stage should proceed back to randomized
  hidden-fault identity or needs another curriculum adjustment.

## Log

- 2026-06-01 Added
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeDeadVelBoost1p6`.
  It wraps the immediate-onset persistent task and fixes
  `left_knee_probability=1.0`, `dynamic_dead_probability=1.0`, and
  `dead_scale_range=(0.0, 0.0)`.
- 2026-06-01 Local focused validation passed:
  `python -m pytest -q -p no:cacheprovider tests\test_task044_hidden_fault_target.py`
  with 6 passed and 1 skipped; `inspect_agent` also passed.
- 2026-06-01 H200 registry patch applied and verified for the new task id.
- 2026-06-01 H200 smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4404701.json`.
- 2026-06-01 H200 25-iteration continuation completed from the
  persistent-hidden checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_persistent_immediate_leftknee_aux002_early4_trial1_scale1_env1024_iter25_seed4404801.json`.
  Checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_persistent_immediate_leftknee_aux002_early4_trial1_scale1_env1024_iter25_seed4404801/model_24.pt`.
- 2026-06-01 H200 triplet eval failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/persistent_immediate_leftknee_aux002_early4_trial1_scale1_iter25_model24_actionstats_triplet_left_knee_joint_vx1p6_seed4404901.json`.
  Normal mode became stable under immediate left-knee dead motor
  (`fall_ratio=0.0`, `gravity_xy.max=0.19454166293144226`,
  `root_z.min=0.7123592495918274`) but too slow
  (`lin_vel_error.mean=1.0232844352722168`).
- 2026-06-01 Intermediate normal probes also stayed slow:
  `model_0` had `lin_vel_error.mean=0.9895840287208557`, and `model_10`
  had `lin_vel_error.mean=0.9683104753494263`.
- 2026-06-01 All-actor update probe was aborted because it destabilized early:
  `fell_over=10.2917` by iteration 2. No pass claim.
- 2026-06-01 Added a speed-push variant
  `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentImmediateLeftKneeDeadSpeedPush1p6`
  with `track_linear_velocity.weight=6.0` and `std=0.5`. Local test and H200
  smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_clear_history_smoke_env64_iter1_seed4405201.json`.
- 2026-06-01 H200 speed-push 10-iteration continuation passed train pipeline:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_persistent_immediate_leftknee_speedpush_aux002_early4_trial1_scale1_env1024_iter10_seed4405301.json`.
  Normal eval failed badly:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_immediate_leftknee_speedpush_iter10_model9_normal_probe_left_knee_joint_vx1p6_seed4405401.json`.
  It had `fall_ratio=1.0`, `gravity_xy.max=0.961262583732605`, and
  `root_z.min=0.18390674889087677`.

## Review

Status: closed as failed curriculum evidence.

The fixed immediate left-knee curriculum solved stability but collapsed speed.
Stronger speed pressure and all-actor updates both destabilized. This subtask
does not pass Task044. The next step should not be more fixed-left-knee training;
it should recover normal quality on the faster persistent-hidden checkpoint
without losing the speed already learned.
