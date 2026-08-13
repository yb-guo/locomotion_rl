# 002 Stage2 Eval

## Route

Run Task050-style recovery eval on the Stage2 checkpoint:

- continuous no-reset eval with `--dynamic-dead-scale 0.3`;
- retry-after-fall eval with `--dynamic-dead-scale 0.3`;
- optional hard Task050 eval with `--dynamic-dead-scale 0.0` for visibility.

Passing Stage2 does not imply hard dead-motor recovery.

## Log

- 2026-08-13 Evaluated the Stage2 checkpoint with stage-matched continuous eval:
  `vx=1.3`, left-knee scale `0.3`, onset `1.0s`. JSON:
  `outputs/task052/eval_stage2/task052_model179_stage2_left_knee_scale0p3_continuous_seed5200201.json`.
  Result: `pass=true`, `pipeline_pass=true`, `quality_gate_pass=true`,
  `physical_continuity_pass=true`, `physical_reset_events=0`,
  `physical_fall_events=0`, post-fault `fall_ratio=0.0`, mean linear velocity
  error `0.097776859998703`, mean yaw velocity error `0.10637470334768295`,
  max `gravity_xy=0.05395696312189102`, and min
  `root_z=0.7685365676879883`.
- 2026-08-13 Evaluated the Stage2 checkpoint with stage-matched retry eval.
  JSON:
  `outputs/task052/eval_stage2/task052_model179_stage2_left_knee_scale0p3_retry_seed5200301.json`.
  Result: `pass=true`, `final_trial_pass=true`, aggregate `fall_count=0`,
  aggregate `fall_ratio=0.0`, and final-trial `fall_count=0`.
- 2026-08-13 Ran the optional hard Task050 migration check:
  `vx=1.6`, left-knee scale `0.0`, onset `0.5s`. Continuous JSON:
  `outputs/task052/eval_hard_task050/task052_model179_hard_left_knee_scale0p0_vx1p6_continuous_seed5200401.json`.
  Result: `pass=false`, `pipeline_pass=false`, `quality_gate_pass=false`,
  `physical_continuity_pass=false`, `physical_reset_events=1034`,
  `physical_fall_events=1034`, post-fault `fall_ratio=1.0`, mean linear
  velocity error `0.8940730094909668`, mean yaw velocity error
  `0.5980390906333923`, max `gravity_xy=0.9500682950019836`, min
  `root_z=0.1703503578901291`, first-fall env ratio `1.0`, and mean first-fall
  time `1.82625s`.
- 2026-08-13 Ran the optional hard Task050 retry check. JSON:
  `outputs/task052/eval_hard_task050/task052_model179_hard_left_knee_scale0p0_vx1p6_retry_seed5200501.json`.
  Result: `pass=false`, `final_trial_pass=false`, aggregate
  `fall_ratio=0.9775390625`, aggregate `fall_count=1001`, final-trial
  `fall_ratio=1.0`, final-trial `fall_count=256`, final-trial mean linear
  velocity error `1.6665771007537842`, max
  `gravity_xy=0.9440926909446716`, and min `root_z=0.1988745629787445`.

## Review

Status: passed for Stage2 partial-fault recovery, failed for optional hard
Task050 dead-motor migration. Proceed to a harder curriculum stage before
claiming hard left-knee dead-motor recovery.
