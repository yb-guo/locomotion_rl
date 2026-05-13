# 006c: Rollout Tilt Sweep Causality

## Goal

Classify the repeated rollout-window tilt reset sweeps seen in subtask006.

The question is narrow: do the sweeps require PPO updates, or does the current
standing setup fall on the same horizon without updates?

## Route

1. Use existing `g1_no_update_ppo_causality`; do not add a new tool unless the
   current evidence is insufficient.
2. H200 physical GPU 1, logical `cuda:0`.
3. Use the task020 standing/reset/action-energy config:
   - `command_mode=standing` through the probe's standing env;
   - `action_scale_mult=0.10`;
   - `log_std_init=-2.0` for sampled-action mode;
   - `termination_height_min=0.20`;
   - `root_z=1.20`;
   - `base_height_reward_scale=0.20`;
   - `termination_penalty=-1.0`.
4. Run no-update modes:
   - `zero_action`;
   - `untrained_mean_action`;
   - `untrained_sampled_action`.
5. Compare first tilt chunk, reset counts, root height, upright, and action
   magnitude to subtask006 PPO gate.

## Acceptance

- Evidence states whether PPO updates are necessary for the reset sweeps.
- If no-update modes reproduce the same horizon, classify as current
  env/contact/passive-standing blocker for this task.
- If only sampled or updated actions reproduce it, classify the next lever as
  action distribution or reward/update control.
- No yaw/vx is introduced.

## Log

- 2026-05-12 Planned after subtask006a showed no single-step full-env reset
  wave, but repeated rollout-window tilt reset sweeps at updates 2, 5, 8, 11,
  14, and 17 for every seed.
- 2026-05-12 Ran three H200 no-update probes through the guarded remote
  protocol with `CUDA_VISIBLE_DEVICES=1`, physical GPU 1, logical `cuda:0`,
  `n_envs=1024`, `chunks=20`, `chunk_steps=32`, `termination_height_min=0.20`,
  `root_z=1.20`, `action_scale_mult=0.10`, `base_height_reward_scale=0.20`,
  and `termination_penalty=-1.0`.
- 2026-05-12 `h200-gpu1-zero-action-v1`: mode `zero_action`, status ok,
  first_tilt_chunk 2, max_reset_count 1024, max_tilt_bad_count 1024,
  mean_reset_count 307.2, final_reset_count 0, final root height mean/min
  0.747360/0.634918, final upright 0.933002, action mean/max 0.0/0.0,
  reset chunks `2,5,8,11,14,17`.
- 2026-05-12 `h200-gpu1-untrained-mean-v1`: mode `untrained_mean_action`,
  status ok, first_tilt_chunk 2, max_reset_count 1024, max_tilt_bad_count
  1024, mean_reset_count 307.2, final_reset_count 0, final root height mean/min
  0.747425/0.635146, final upright 0.933145, action mean/max
  0.002260/0.011355, reset chunks `2,5,8,11,14,17`.
- 2026-05-12 `h200-gpu1-untrained-sampled-v1`: mode
  `untrained_sampled_action`, status ok, first_tilt_chunk 2, max_reset_count
  1024, max_tilt_bad_count 1024, mean_reset_count 307.2, final_reset_count 0,
  final root height mean/min 0.741173/0.523199, final upright 0.922823, action
  mean/max 0.106711/0.573839, reset chunks `2,5,8,11,14,17`.
- 2026-05-12 Interpretation: PPO updates are not necessary for the observed
  reset sweeps. The current standing setup falls on the same chunk horizon with
  zero normalized action, tiny untrained mean action, and sampled untrained
  action. This classifies the task020 standing gate blocker as current
  env/contact/passive-standing dynamics, not PPO plumbing or action-energy
  tuning.

## Review

Status: pending evidence review.
