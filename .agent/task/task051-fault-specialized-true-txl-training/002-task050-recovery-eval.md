# 002 Task050 Recovery Eval

## Route

Run the Task050 recovery eval pair on the Task051-trained checkpoint:

- continuous no-reset hidden left-knee dead-motor eval;
- retry-after-fall hidden left-knee dead-motor eval.

Both evals must use the same checkpoint and record JSON evidence. Passing the
training smoke alone is not a recovery claim.

## Log

- 2026-08-12 Evaluated the 10-iteration Task051 smoke checkpoint with Task050
  continuous eval. JSON:
  `outputs/task051/eval_left_knee_fault_train/task051_model9_task050_left_knee_dead_continuous_seed5100201.json`.
  Result: `pipeline_pass=false`, `quality_gate_pass=false`,
  `physical_continuity_pass=false`, `physical_reset_events=512`,
  `physical_fall_events=512`, post-fault `fall_ratio=1.0`.
- 2026-08-12 Evaluated the 10-iteration Task051 smoke checkpoint with Task050
  retry eval using the matching True-TXL actor config. JSON:
  `outputs/task051/eval_left_knee_fault_train/task051_model9_task050_left_knee_dead_retry_actorcfg_seed5100301.json`.
  Result: `pass=false`, `final_trial_pass=false`; final trial had
  `fall_ratio=1.0`, `fall_count=256`, mean linear velocity error
  `1.6433457136154175`, max `gravity_xy=0.9392699003219604`, and min
  `root_z=0.26390576362609863`.
- 2026-08-12 Evaluated the 90-iteration Task051 continuation checkpoint with
  Task050 continuous eval. JSON:
  `outputs/task051/eval_left_knee_fault_train/task051_model89_task050_left_knee_dead_continuous_seed5100202.json`.
  Result: `pipeline_pass=false`, `quality_gate_pass=false`,
  `physical_continuity_pass=false`, `physical_reset_events=512`,
  `physical_fall_events=512`, post-fault `fall_ratio=1.0`, mean linear
  velocity error `0.7162381410598755`, mean yaw velocity error
  `0.41009289026260376`, max `gravity_xy=0.9545283317565918`, and min
  `root_z=0.30517569184303284`.
- 2026-08-12 Evaluated the 90-iteration Task051 continuation checkpoint with
  Task050 retry eval using the matching True-TXL actor config
  (`--memory-latent-dim 32`, `--action-dim 31`,
  `--base-obs-passthrough`, `--adaptation-warmstart`). JSON:
  `outputs/task051/eval_left_knee_fault_train/task051_model89_task050_left_knee_dead_retry_actorcfg_seed5100302.json`.
  Result: `pass=false`, `final_trial_pass=false`, aggregate
  `fall_ratio=1.0`, aggregate `fall_count=1024`. Final trial had
  `fall_ratio=1.0`, `fall_count=256`, mean linear velocity error
  `1.5924187898635864`, mean yaw velocity error `0.7879140973091125`, max
  `gravity_xy=0.9451430439949036`, and min `root_z=0.3090360760688782`.

## Review

Status: failed for the current Task051 candidates. The Task050 continuous
physical-continuity gate and retry final-trial promotion gate both reject the
fault-specialized checkpoints.
