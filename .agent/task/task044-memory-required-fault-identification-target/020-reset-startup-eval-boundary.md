# 020: Reset Startup Eval Boundary

## Route

Subtasks 018 and 019 failed to close the original 2.0 s strict normal eval by
adding more speed pressure. The velocity-component and tail-window diagnostics
show a different boundary:

- the policy can reach and hold the commanded 1.6 m/s in the final-trial tail;
- the strict full-final metric is dominated by the first post-reset startup;
- Task037/MJLab inner reset restores the hidden fault/command condition but
  resets the robot physical state to deterministic standing and near-zero
  velocity;
- `Task044TrueTxlMemoryK160ClearHistoryRunner` clears actor-visible K160 history
  on inner reset while preserving the TXL cache, so the memory signal can
  survive but the body still has to accelerate from rest.

This subtask records that boundary before changing either the quality gate or
the runner contract.

## Acceptance

- H200 JSON evidence must show the original 2.0 s full-final strict gate still
  fails.
- The same checkpoint must show substantially better last-window/tail speed.
- A longer-trial diagnostic must show whether final-threshold locomotion can
  pass once startup is less dominant.
- No Task044 pass claim is allowed from tail or longer-trial evidence alone;
  the original 2.0 s full-final gate remains failed unless explicitly changed
  by a later route.

## Log

- 2026-06-01 Re-eval of the best pose-tight checkpoint with velocity
  components and first/tail windows:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_pose_tight_lr1e5_from_pose_iter10_model9_normal_probe_left_knee_joint_vx1p6_seed4415101_tail.json`.
  Full final trial failed with `lin_vel_error.mean=0.4931211471557617` and
  `lin_vel_actual.mean_x=1.2648766040802002`. The first 0.5 s final window had
  `lin_vel_actual.mean_x=0.38676440715789795` and
  `lin_vel_error.mean=1.214076042175293`. The last 0.5 s tail window had
  `lin_vel_actual.mean_x=1.660741925239563` and
  `lin_vel_error.mean=0.2633638083934784`.
- 2026-06-01 A 3.0 s trial diagnostic on the same checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/persistent_hidden_pose_tight_lr1e5_from_pose_iter10_model9_normal_probe_left_knee_joint_vx1p6_trial3s_seed4415701_tail.json`.
  The full final trial had `lin_vel_error.mean=0.4277479946613312`,
  `lin_vel_actual.mean_x=1.444486141204834`, `fall_ratio=0.06521739065647125`,
  `gravity_xy.max=0.7459681034088135`, and `root_z.min=0.571283221244812`.
  Final thresholds passed, but the diagnostic quality gate remained false due
  to trend regressions (`root_z_min_regressed_from_trial0` and
  `yaw_vel_error_mean_regressed_from_trial0`).
- 2026-06-01 Code inspection confirmed the reset boundary:
  `src/h200_locomotion_lab/training/mjlab_inner_reset.py` calls the original
  MJLab reset for inner resets and only restores command/fault condition
  tensors; it does not preserve physical robot velocity or gait phase.
  `src/h200_locomotion_lab/training/rsl_history_wrapper.py` then clears
  actor-visible history on inner reset for the Task044 runner.
- 2026-06-01 Added `final_trial_tail_window` as an allowed metric scope in the
  Task044 triplet-summary CLI and verified it locally:
  `13 passed` for
  `tests/test_task044_memory_required_contract.py` and
  `tests/test_task044_triplet_summary_cli.py`. CLI `--help` also shows
  `final_trial_tail_window`.
- 2026-06-01 H200 tail-scope triplet summary recorded:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/persistent_hidden_pose_tight_model9_tail_scope_triplet_seed4415101.json`.
  Result: `task044_memory_required_pass=false` with
  `normal_quality_gate_not_passed`, `zero_residual_ablation_not_degraded`, and
  `stateless_memory_ablation_not_degraded`. Tail metrics did not prove memory
  causality: normal `lin_vel_error.mean=0.2633638083934784`, zero-residual
  `0.2583812177181244` (`delta=-0.004982590675354004`), and stateless
  `0.2517612874507904` (`delta=-0.011602520942687988`).

## Review

Status: diagnostic closed, not passed.

The current blocker has two parts. First, the original 2.0 s full-final gate is
dominated by post-reset acceleration from deterministic standing. Second, even
when using a tail-only metric scope, the ablations remain tied or slightly
better than normal, so this checkpoint still does not prove memory-required
control. A later route must choose between preserving the original strict
startup requirement, changing the eval metric to a steady-state/tail scope, or
changing the runner/reset contract/policy objective so TXL memory is actually
needed and used.
