# 021: Continuous Hidden Fault No-Reset Eval

## Route

Subtask 020 showed two separate blockers:

- the original 2.0 s full-final gate is polluted by deterministic standing
  startup after each inner physical reset;
- tail-scope normal/zero-residual/stateless metrics remain tied, so the current
  checkpoint still does not prove memory-required control.

This subtask adds an eval-only route that removes the physical reset confound.
It runs one continuous episode, starts clean, injects a hidden dynamic motor
fault mid-episode, excludes a short post-onset transient, and evaluates a
`post_fault_window`. It also adds a stronger `zero_memory_latent` ablation that
zeros the combined TXL residual plus adaptation-history latent while preserving
the newest base observation passthrough.

## Acceptance

- The continuous eval runner must not install `Task037MultiTrialVecEnvWrapper`
  or `install_task037_inner_reset_controller`.
- The eval JSON must record:
  - `task044_continuous_fault_eval=true`;
  - `startup_excluded_s`;
  - `post_fault_window`;
  - `physical_reset_events`;
  - `inner_reset_events_total`;
  - hidden-fault actor-observation contract fields;
  - no quality, eval, reproduction, superiority, or memory-causality claim.
- `post_fault_window` must be selectable as a Task044 triplet metric scope.
- `zero_memory_latent` must be accepted as a stronger zero-side ablation by the
  Task044 triplet contract.
- H200 must produce normal / zero-memory-latent / stateless JSONs from the same
  checkpoint, seed, command, and fault schedule.
- Task044 is not passed unless normal `post_fault_window` quality passes and
  both ablations materially degrade under the triplet summary.

## Log

- 2026-06-01 Added local contract/code:
  - `Task044TrueTxlMemoryK160ContinuousRunner`;
  - `h200_locomotion_lab.tools.task044_continuous_fault_eval`;
  - `post_fault_window` triplet metric scope;
  - `zero_memory_latent` ablation mode.
- 2026-06-01 Local validation passed:
  `31 passed` for continuous eval, Task042 ablation, Task044 contract, and
  triplet-summary tests. H200 targeted validation also passed with `31 passed`;
  CLI `--help` shows `zero_memory_latent` and the continuous/post-fault args.
- 2026-06-01 H200 continuous normal eval recorded:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/continuous_fault_eval/pose_tight_model9_continuous_normal_left_knee_seed4417001.json`.
  It had `inner_reset_events_total=0`, proving the Task037 physical inner reset
  path was removed. It still failed: `physical_reset_events=57`,
  `post_fault_window.fall_ratio=0.12109375`,
  `post_fault_window.lin_vel_error.mean=0.2691524028778076`,
  `post_fault_window.lin_vel_actual.mean_x=1.6762081384658813`,
  `post_fault_window.gravity_xy.max=0.7421531081199646`, and
  `post_fault_window.root_z.min=0.5708710551261902`.
- 2026-06-01 H200 continuous post-fault triplet summary recorded:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/pose_tight_model9_continuous_post_fault_triplet_seed4417001.json`.
  Result: `task044_memory_required_pass=false` with
  `normal_pipeline_not_passed`, `normal_quality_gate_not_passed`,
  `zero_residual_ablation_not_degraded`, and
  `stateless_memory_ablation_not_degraded`.
  `zero_memory_latent` had `policy_memory_latent_norm=0.0` and
  `memory_latent_enabled=false`, but it did not degrade behavior:
  post-fault `lin_vel_error.mean=0.25023153424263` and
  `fall_ratio=0.09375`, both slightly better than normal. Stateless TXL was
  also not materially worse: `lin_vel_error_delta=0.014190614223480225` and
  `fall_ratio_delta=0.015625`, below Task044 degradation thresholds.

## Review

Status: diagnostic closed, not passed.

This subtask is the next diagnostic gate, not a LocoFormer reproduction claim.
It successfully separates hidden-fault adaptation from deterministic standing
startup, but the current checkpoint still fails normal continuous quality due
to physical resets and still does not use the memory latent in a behaviorally
necessary way. The next route should change the policy consumer/training target
so current-observation/base-observation passthrough cannot solve the task
without history.
