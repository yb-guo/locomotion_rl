# Task 054: Longer Sampled Fault Continuation

## Route

Continue the Task053 sampled hidden left-knee fault curriculum after the first
hard-gate failure. Use the Task053 180-iteration checkpoint as warm start and
run a longer continuation:

- source checkpoint:
  `outputs/task053/sampled_curriculum/task053_sampled_curriculum_env512_step24_iter180_mb1_seed5300102/model_179.pt`;
- train task:
  `Unitree-G1-Gripper-Flat-Task053-TrueTxl-SampledCurriculum-Train`;
- actor boundary remains hidden: no explicit fault identity, scale, onset, or
  recovery label is visible to the actor;
- run enough iterations that the curriculum re-enters full-hard sampling and
  spends substantial updates there.

## Acceptance Criteria

- Continuation train JSON exists with `train_pipeline_pass=true` and a final
  checkpoint.
- Hard Task050 continuous eval JSON exists for `vx=1.6`, left-knee
  `scale=0.0`, onset `0.5s`.
- Hard Task050 retry eval JSON exists for the same hard setting.
- No hard recovery claim is made unless both hard gates pass.
- No all-joint damaged-joint claim is made from the left-knee-only target.

## Log

- 2026-08-13 Opened after Task053 180-iteration checkpoint trained but failed
  both hard Task050 continuous and retry gates. User asked whether to run more.
- 2026-08-13 Ran 360-iteration continuation from the Task053 180-iteration
  checkpoint:
  `.agent/task/task054-longer-sampled-fault-continuation/task054_sampled_continuation_env512_step24_iter360_mb1_seed5400101.json`.
  Training passed with `train_pipeline_pass=true`,
  `task053_train_pipeline_pass=true`, `checkpoint_exists=true`,
  `final_iteration=359`, and final checkpoint
  `outputs/task054/sampled_continuation/task054_sampled_continuation_env512_step24_iter360_mb1_seed5400101/model_359.pt`.
  The run crossed the full-hard sampled curriculum point at iteration 180 and
  continued through iteration 359.
- 2026-08-13 Ran hard Task050 continuous eval on `model_359.pt` with
  `vx=1.6`, hidden dynamic `left_knee_joint` failure, `scale=0.0`, onset
  `0.5s`, recovery `999.0s`:
  `outputs/task054/eval_hard_task050/task054_model359_sampled_hard_left_knee_scale0p0_vx1p6_continuous_seed5400201.json`.
  Result: `pass=false`, `physical_continuity_pass=false`,
  `quality_gate_pass=false`, `physical_reset_events=129`,
  `physical_fall_events=129`, aggregate `fall_ratio=0.50390625`,
  `lin_vel_error.mean=0.8476555943489075`,
  `yaw_vel_error.mean=0.37522605061531067`, `gravity_xy.max=0.9528967142105103`,
  `root_z.min=0.09008878469467163`. Failure reasons:
  `physical_continuity_not_preserved`, `post_fault_window_quality_not_passed`,
  `post_fault_lin_vel_error_too_high`.
- 2026-08-13 Ran matching hard Task050 retry eval on `model_359.pt`:
  `outputs/task054/eval_hard_task050/task054_model359_sampled_hard_left_knee_scale0p0_vx1p6_retry_seed5400301.json`.
  Result: `pass=false`, `final_trial_pass=false`. Aggregate:
  `fall_count=0`, `fall_ratio=0.0`, `lin_vel_error_mean=1.190990686416626`,
  `yaw_vel_error_mean=0.24990993738174438`,
  `gravity_xy_max=0.18121124804019928`, `root_z_min=0.6805713176727295`.
  Final trial: `fall_count=0`, `fall_ratio=0.0`,
  `lin_vel_error.mean=1.2770731449127197`,
  `yaw_vel_error.mean=0.2532256543636322`,
  `gravity_xy.max=0.1487695425748825`,
  `root_z.min=0.7542341947555542`.

## Review

Status: complete, negative hard-recovery evidence.

The longer sampled curriculum continuation produced a valid final checkpoint and
improved physical survival relative to the failed Task053 hard eval, but it did
not satisfy the Task050 hard recovery gates. The continuous gate still had 129
physical fall/reset events and excessive post-fault linear velocity error. The
retry gate avoided falls, including in the final trial, but failed because the
policy moved far below the commanded `vx=1.6` target. No hard recovery claim is
made.
