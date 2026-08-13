# 002 Hard Task050 Eval

## Route

Run the hard Task050 recovery eval pair on the sampled-curriculum checkpoint:

- continuous no-reset hidden left-knee dead-motor eval at `vx=1.6`, scale
  `0.0`, onset `0.5s`;
- retry-after-fall hidden left-knee dead-motor eval with the same hard setting.

Passing sampled training alone is not a hard recovery claim.

## Log

- 2026-08-13 Evaluated Task053 180-iteration sampled-curriculum checkpoint:
  `outputs/task053/sampled_curriculum/task053_sampled_curriculum_env512_step24_iter180_mb1_seed5300102/model_179.pt`.
- 2026-08-13 Hard continuous no-reset eval JSON:
  `outputs/task053/eval_hard_task050/task053_model179_sampled_hard_left_knee_scale0p0_vx1p6_continuous_seed5300201.json`.
  Result: `pass=false`, `physical_continuity_pass=false`,
  `quality_gate_pass=false`, `physical_reset_events=768`,
  `physical_fall_events=768`, post-fault `fall_ratio=1.0`,
  `lin_vel_error.mean=0.7989295125007629`,
  `yaw_vel_error.mean=0.5508509874343872`,
  `gravity_xy.max=0.9580077528953552`, and
  `root_z.min=0.2514921724796295`.
- 2026-08-13 Hard retry eval JSON:
  `outputs/task053/eval_hard_task050/task053_model179_sampled_hard_left_knee_scale0p0_vx1p6_retry_seed5300301.json`.
  Result: `pass=false`, `final_trial_pass=false`, aggregate
  `fall_count=1024`; final trial `fall_count=256`, `fall_ratio=1.0`,
  `lin_vel_error.mean=1.6858710050582886`,
  `yaw_vel_error.mean=0.6042336821556091`,
  `gravity_xy.max=0.9432367086410522`, and
  `root_z.min=0.32019105553627014`.

## Review

Status: complete, failed hard Task050 gates.

The one-shot sampled curriculum trains and produces checkpoints, but the
180-iteration candidate does not demonstrate hidden left-knee dead-motor
recovery. Continuous recovery fails due to physical falls/resets and post-fault
quality violations. Retry-after-fall recovery also fails because the final
trial still falls in all 256 envs. This is negative hard-gate evidence for this
checkpoint only, not an all-joint damaged-joint claim.
