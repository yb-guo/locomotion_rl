# 017: Root-Height Guard Stage

## Route

Persistent-hidden velocity tuning reached a narrower failure mode: the policy
can improve `lin_vel_error`, but it does so by accepting low-root outliers.
This subtask adds root-height guard variants without changing actor-visible
obs, action shape, failure schedule semantics, or the Task044 triplet gate.

The guard variants are:

- `PersistentHiddenHeightGuard1p6`: inherits the speed-stability task and adds
  `base_height_below_l2` with `min_height=0.70`, `weight=-8.0`.
- `PersistentHiddenHeightGuardStrong1p6`: additionally sets
  `track_linear_velocity.weight=5.0`, `std=0.7`, and strengthens
  `base_height_below_l2` to `min_height=0.72`, `weight=-24.0`.

## Acceptance

- Local tests lock the new reward patch, task ids, and train/eval allowlists.
- H200 registry patch writes both task ids and the reward function.
- H200 smoke must pass before any long train.
- H200 eval must report both locomotion quality and failure reasons.
- No pass claim is allowed unless normal quality passes and the full Task044
  triplet also proves memory ablation degradation.

## Log

- 2026-06-01 Local validation passed:
  `17 passed, 7 skipped` for
  `tests/test_task044_hidden_fault_target.py` and
  `tests/test_task041_sequence_txl_clean_train.py`; `inspect_agent` passed.
- 2026-06-01 H200 `PersistentHiddenHeightGuard1p6` smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_persistent_hidden_height_guard_smoke_env64_iter1_seed4409001.json`.
- 2026-06-01 Weak height guard from the current best checkpoint produced the
  closest normal eval but still failed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_height_guard_lr1e5_from_best_iter30_model29_normal_probe_left_knee_joint_vx1p6_seed4409401.json`
  with final `lin_vel_error.mean=0.5106924772262573`,
  `root_z.min=0.5476866960525513`, and
  `gravity_xy.max=0.7159021496772766`.
- 2026-06-01 Strong height guard smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/train_persistent_hidden_height_guard_strong_smoke_env64_iter1_seed4409501.json`.
- 2026-06-01 Strong height guard continuations did not close the gap. LR1e-5
  eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_height_guard_strong_lr1e5_from_weakguard_iter20_model19_normal_probe_left_knee_joint_vx1p6_seed4409801.json`
  failed with `root_z.min=0.23591351509094238` and
  `lin_vel_error.mean=0.504571795463562`. LR5e-6 eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_height_guard_strong_lr5e6_from_weakguard_iter20_model19_normal_probe_left_knee_joint_vx1p6_seed4409901.json`
  failed with `root_z.min=0.35728371143341064` and
  `lin_vel_error.mean=0.5264725089073181`.

## Review

Status: diagnostic closed, not passed.

The ordinary height reward helped but did not solve the gate. The best weak
height-guard checkpoint missed the strict root threshold by about `0.0023` and
the velocity threshold by about `0.061`. Stronger average reward did not fix
the outlier root problem. The next repair should be a hard low-root termination
or reset/penalty guard, not more average reward scaling.
