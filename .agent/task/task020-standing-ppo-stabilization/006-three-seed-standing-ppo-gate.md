# 006: Three Seed Standing PPO Gate

## Goal

Run the standing PPO stabilization gate with selected config.

## Route

1. H200 physical GPU 1.
2. 3 seeds.
3. `command_mode=standing`.
4. Suggested gate:
   - `n_envs=1024`;
   - `rollout_steps=32`;
   - `ppo_updates=20`;
   - `epochs=2`;
   - `minibatch_size=8192`.
5. Record train metrics and summary.

## Acceptance

- All success metrics in task contract pass.
- No NaN/Inf.
- Actor/value params change.
- No full-env reset wave every rollout.
- Episode length reaches `>= 200` or `>= 2x` baseline.
- Readiness for deterministic standing eval is clear.

## Log

- 2026-05-12 Planned.
- 2026-05-12 Ran H200 gate `h200-gpu1-standing-gate-v1` through the guarded
  remote protocol with `CUDA_VISIBLE_DEVICES=1`, physical GPU 1, logical
  `cuda:0`, 3 seeds, `command_mode=standing`, `ppo_updates=20`, `n_envs=1024`,
  `rollout_steps=32`, `epochs=2`, and `minibatch_size=8192`.
- 2026-05-12 Config used the subtask005 selected action-energy candidate and
  subtask004 reward/reset pack: `action_scale_mult=0.10`,
  `log_std_init=-2.0`, `base_height_reward_scale=0.20`,
  `joint_velocity_penalty_scale=0.001`, `termination_penalty=-1.0`,
  `termination_height_min=0.20`, and `root_z=1.20`.
- 2026-05-12 Run result: `status=ok`, `all_seeds_passed=true`,
  `mean_final_survival_rate=1.0`, max final height/tilt/timeout reset rates all
  0.0, no final full-env reset wave, and min collect throughput 44411.12
  env-policy steps/s. Actor and value parameters changed for all 3 seeds.
- 2026-05-12 Gate failure: final `mean_final_episode_length_mean=67.295247`.
  The subtask003 baseline was 51.9209, so the task contract's easier threshold
  is about 103.84 (`2x` baseline). Per-seed final episode means were 66.956,
  67.403, and 67.526, with per-seed maximum episode means during training only
  about 71.34, 71.38, and 71.26.
- 2026-05-12 Additional signal: reward peaked early near 2.216 per seed but
  ended at 1.502876, 1.488745, and 1.480386. Root height also ended low
  (`root_height_mean` 0.745323, 0.742941, 0.741399; `root_height_min` 0.550936,
  0.528411, 0.512587). This indicates the selected low-energy PPO run avoids
  hard resets but does not meet the standing-stabilization episode-length gate.
  Artifacts copied locally under
  `.agent/task/task020-standing-ppo-stabilization/artifacts/h200-gpu1-standing-gate-v1/`.
- 2026-05-12 After subtask006a added training-wide reset metrics, re-ran the
  same gate as `h200-gpu1-standing-gate-v2`. Result stayed blocked:
  `mean_final_episode_length_mean=67.291992`, max training episode length
  71.383545, min collect throughput 44088.53 env-policy steps/s, and all
  plumbing checks passed. There were no single-step full-env reset waves
  (`training_full_env_reset_wave_count=0`), but every seed had rollout-window
  tilt reset sweeps at updates 2, 5, 8, 11, 14, and 17
  (`reset_count=1024`, `tilt_reset_count=1024`, max reset/tilt reset rate
  0.03125). Artifacts copied locally under
  `.agent/task/task020-standing-ppo-stabilization/artifacts/h200-gpu1-standing-gate-v2/`.

## Review

Status: complete. Training plumbing passed, but the standing PPO gate did not
meet the episode-length acceptance threshold and showed repeated rollout-window
tilt reset sweeps.
