# 009: Forward Speed Expansion

## Route

Extend the accepted task029 BalancedKnee policy toward faster forward walking
without changing the robot topology, actor/action contract, reward stack, or
motor-failure information boundary.

The first speed target is fixed-command `lin_vel_x=1.0 m/s`. After it closes,
the same pattern can advance to `1.2 m/s` and above.

Keep:

- Fixed G1-like gripper topology.
- Fixed link geometry, mass, COM, and inertia.
- Existing MLP PPO baseline.
- Existing episode-start persistent motor-failure sampler.
- Balanced single-dead left/right knee oversampling.
- No actor-visible fault labels, motor scales, or failure masks.

Change:

- Add forward-speed task variants that keep BalancedKnee but narrow the command
  distribution to positive forward speeds:
  - `Fast1p0`: `lin_vel_x=(0.4, 1.0)`.
  - `Fast1p2`: `lin_vel_x=(0.6, 1.2)`.
  - `Fast1p4`: `lin_vel_x=(0.8, 1.4)`.
- For these forward-speed variants, set `lin_vel_y=(0.0, 0.0)`,
  `ang_vel_z=(0.0, 0.0)`, disable heading command, and disable standing envs.

## Minimal Closed Loop

Feedback loop:

1. Register forward-speed task variants in the H200 MJLab checkout.
2. Inspect task contract: actor/action dims unchanged, motor-only events only,
   no actor fault-label leak, command range matches the speed stage.
3. Run a 64-env PPO smoke for `Fast1p0`.
4. Train `Fast1p0` on H200, preferably resuming from the accepted BalancedKnee
   checkpoint.
5. Evaluate fixed commands at `0.5`, `0.8`, and `1.0 m/s`; include at least
   clean, in-distribution failure, doubled holdout, and selected knee-dead
   checks before accepting.
6. Render `1.0 m/s` clean and left-knee-dead videos for visual review.
7. Only after `1.0 m/s` passes, repeat the same loop for `1.2 m/s`.

Pass for `Fast1p0`:

- `action_dim=31`, `actor_obs=104`.
- No actor motor/failure/fault/scale labels.
- Clean fixed `1.0 m/s` eval has no falls and acceptable tracking.
- Left/right knee-dead checks still pass at `1.0 m/s`.
- Render does not show obvious reward hacking, foot dragging, or violent
  upper-body motion.

Fail:

- The faster policy only survives by ignoring the speed command.
- Clean `0.5 m/s` regresses severely.
- Left/right knee-dead robustness regresses.
- The task is marked passed without H200 eval and render evidence.

Evidence root:

`/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/`.

## Log

- 2026-05-20 Opened after the accepted BalancedKnee checkpoint was probed with
  fixed `0.8 m/s`. It did not fall, but clean fixed-command tracking was poor:
  `lin_vel_error_mean=0.5269265174865723`, so high speed needs training
  distribution work rather than only changing eval/render command.
- 2026-05-20 Registered and inspected the forward-speed task variants on H200.
  Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/inspect/task029_forward_speed_inspect_summary.json`.
  `Fast1p0`, `Fast1p2`, and `Fast1p4` preserved `action_dim=31`,
  `actor_obs=104`, `critic_obs=119`, and had no actor-visible motor/failure
  labels.
- 2026-05-20 `Fast1p0` resumed from the accepted BalancedKnee checkpoint and
  trained to:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/2026-05-20_15-05-01_009_fast1p0_resume_env8192_iter800_gpu1_seed29120/model_1998.pt`.
  Clean fixed `1.0 m/s` tracking passed, but the full dead-motor grid failed
  on left/right hip-yaw forced-dead cases.
- 2026-05-20 BalancedCritical and BalancedRightCritical continuations showed a
  side-specific forgetting pattern: one side's hip-yaw/knee failures could be
  fixed while the opposite side regressed. The stable fix was the
  PhaseBalancedCritical `Fast1p0` continuation from `model_2797.pt`.
- 2026-05-20 Accepted `Fast1p0` checkpoint for fixed `1.0 m/s`:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/2026-05-20_17-33-37_009_phasebalancedcritical_fast1p0_resume2797b_env8192_iter800_gpu0_seed29153/model_3100.pt`.
  Full eval evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p0_eval_phasecritical_model3100_v1p0_full_grid/task029_eval_failure_aggregate.json`.
  Aggregate `pass=true`, `complete_grid=true`, `grid_case_count=12`,
  `clean_motor_failure_stripped=true`, and fixed command
  `lin_vel_x=1.0`.
- 2026-05-20 `model_3100.pt` fixed `1.0 m/s` metrics: clean
  `zero_fall_ratio=1.0`, `max_done_count=0`,
  `lin_vel_error_mean=0.10559709370136261`; doubled holdout passed with
  `zero_fall_ratio=0.9375`; all 12 forced-dead grid cases passed. Critical
  repeated validation also passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p0_eval_phasecritical_model3100_v1p0_critical_multiseed_s5/task029_dead_motor_knee_multiseed_s5_summary.json`.
- 2026-05-20 `model_3100.pt` fixed `1.0 m/s` render evidence passed for clean,
  in-distribution failure, and forced `right_hip_yaw_joint` dead:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/render_phasecritical_model3100_v1p0_right_hip_yaw/task029-render-phasecritical-model3100-v1p0-summary.json`.
  A local copy for review is under
  `outputs/task029/forward_speed_expansion/render_phasecritical_model3100_v1p0_right_hip_yaw/`.
- 2026-05-20 Started `Fast1p2` after `Fast1p0` closed. A 64-env resume smoke
  for `Unitree-G1-Gripper-Flat-MotorOnly-Failure-PhaseBalancedCritical-Fast1p2`
  loaded `model_3100.pt` and ran two iterations with zero falls. Two H200
  full-training seeds are running from the same checkpoint:
  `009_phasebalancedcritical_fast1p2_resume3100_env8192_iter600_gpu0_seed29161`
  and
  `009_phasebalancedcritical_fast1p2_resume3100_env8192_iter600_gpu1_seed29162`.
- 2026-05-20 `Fast1p2` needed several narrow continuations because full-grid
  robustness traded off between hip-yaw and knee failures. The accepted fixed
  `1.2 m/s` checkpoint is:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/2026-05-20_20-46-25_009_phaseyawkneequad_fast1p2_resume3650_env8192_iter75_gpu0_seed29200/model_3675.pt`.
  It was produced by the four-joint PhaseYawKneeQuad stage, oversampling
  `left_hip_yaw_joint`, `right_hip_yaw_joint`, `left_knee_joint`, and
  `right_knee_joint`.
- 2026-05-20 `Fast1p2` full-grid evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p2_eval_phaseyawkneequad_seed29200_model3675_v1p2_full_grid/task029_eval_failure_aggregate.json`.
  Aggregate `pass=true`, `complete_grid=true`, `grid_case_count=12`,
  `clean_motor_failure_stripped=true`, fixed `lin_vel_x=1.2`, clean
  `zero_fall_ratio=1.0`, clean `lin_vel_error_mean=0.0951637402176857`,
  doubled holdout `zero_fall_ratio=0.9375`, and all 12 forced-dead grid cases
  passed. The thinnest margin was `right_knee_joint` forced-dead with
  `zero_fall_ratio=0.5234375`.
- 2026-05-20 `Fast1p2` critical repeated validation passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p2_eval_phaseyawkneequad_seed29200_model3675_v1p2_critical_multiseed_s5/task029_dead_motor_knee_multiseed_s5_summary.json`.
  Five seeds passed for left/right hip-yaw and left/right knee. `right_knee`
  remained the limiting case with zero-fall ratios
  `[0.56640625, 0.51953125, 0.55859375, 0.58203125, 0.515625]`.
- 2026-05-20 `Fast1p2` render evidence passed for clean, in-distribution
  failure, and forced `right_knee_joint` dead:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/render_phaseyawkneequad_model3675_v1p2_right_knee/task029-render-phaseyawkneequad-model3675-v1p2-summary.json`.
  Local review copies are under
  `outputs/task029/forward_speed_expansion/render_phaseyawkneequad_model3675_v1p2_right_knee/`.
- 2026-05-20 Registered and inspected `PhaseYawKneeQuad-Fast1p4` with
  `lin_vel_x=(0.8, 1.4)`. Contract evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/inspect_yawkneequad/task029_forward_speed_yawkneequad_inspect_summary.json`.
  A 64-env smoke from the accepted `Fast1p2 model_3675.pt` passed, then two
  `Fast1p4` 8192-env training seeds were launched:
  `009_phaseyawkneequad_fast1p4_resume3675_env8192_iter300_gpu0_seed29210`
  and
  `009_phaseyawkneequad_fast1p4_resume3675_env8192_iter300_gpu1_seed29211`.
- 2026-05-20 `PhaseYawKneeQuad-Fast1p4` trained to `model_3974.pt` on both
  seeds, but critical fixed `1.4 m/s` repeated eval failed only
  `left_knee_joint` forced-dead. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p4_eval_phaseyawkneequad_seed29210_model3974_v1p4_critical_multiseed_s3/task029_dead_motor_knee_multiseed_s5_summary.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p4_eval_phaseyawkneequad_seed29211_model3974_v1p4_critical_multiseed_s3/task029_dead_motor_knee_multiseed_s5_summary.json`.
  Hip-yaw guard joints passed, right knee passed, and left knee had
  zero-fall ratios around `0.30-0.43`.
- 2026-05-20 Earlier `Fast1p4` checkpoints did not provide a natural
  acceptance point for the left-knee failure. `model_3950.pt` left-knee-only
  critical s3 had zero-fall ratios around `0.34-0.40`; `model_3900.pt`
  regressed further to about `0.21-0.24`.
- 2026-05-20 Registered and inspected
  `PhaseLeftKneeGuard-Fast1p4`, a narrow continuation task that keeps
  topology, reward, action, and actor observation unchanged but oversamples
  `left_knee_joint` dead-motor episodes while retaining `right_knee_joint` and
  both hip-yaw joints as guards. Contract evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/inspect_leftkneeguard/task029_forward_speed_leftkneeguard_inspect_summary.json`.
  Inspect passed with `action_dim=31`, `actor_obs=104`, no actor fault-label
  leak, `lin_vel_x=(0.8, 1.4)`, and left-knee reset oversampling.
- 2026-05-20 A 64-env smoke for `PhaseLeftKneeGuard-Fast1p4` resumed from
  `PhaseYawKneeQuad-Fast1p4` GPU1 `model_3974.pt` and completed. Two 8192-env
  continuation seeds were launched from the same checkpoint:
  `009_phaseleftkneeguard_fast1p4_resume3974_env8192_iter200_gpu0_seed29220`
  and
  `009_phaseleftkneeguard_fast1p4_resume3974_env8192_iter200_gpu1_seed29221`.
- 2026-05-20 `PhaseLeftKneeGuard-Fast1p4` GPU1 `model_4173.pt` passed the
  fixed `1.4 m/s` four-joint critical s3 gate for left/right hip-yaw and
  left/right knee. Left knee improved to zero-fall ratios
  `[0.9453125, 0.9453125, 0.953125]`. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p4_eval_phaseleftkneeguard_seed29221_model4173_v1p4_critical_multiseed_s3/task029_dead_motor_knee_multiseed_s5_summary.json`.
- 2026-05-20 The same `model_4173.pt` did not pass full-grid at fixed
  `1.4 m/s`. Clean, motor-primitives, in-distribution failure, doubled
  holdout, left knee, and right knee passed; failures moved to hip pitch/roll:
  `left_hip_pitch_joint`, `left_hip_roll_joint`, and
  `right_hip_roll_joint`. Full-grid evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p4_eval_phaseleftkneeguard_seed29221_model4173_v1p4_full_grid/task029_eval_failure_aggregate.json`.
  The alternate GPU0 `model_4173.pt` was not better: a targeted critical s3
  still failed left/right hip-roll and regressed left knee.
- 2026-05-20 Registered and inspected
  `PhaseHipRollPitchGuard-Fast1p4`, a second narrow continuation that
  oversamples right/left hip-roll, left/right hip-pitch, and keeps knee/yaw
  guards. Contract evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/inspect_hiprollpitchguard/task029_forward_speed_hiprollpitchguard_inspect_summary.json`.
  A 64-env resume smoke from `PhaseLeftKneeGuard-Fast1p4` GPU1
  `model_4173.pt` completed. Two 8192-env continuation seeds were launched:
  `009_phasehiprollpitchguard_fast1p4_resume4173_env8192_iter250_gpu0_seed29230`
  and
  `009_phasehiprollpitchguard_fast1p4_resume4173_env8192_iter250_gpu1_seed29231`.
- 2026-05-20 `PhaseHipRollPitchGuard-Fast1p4` final checkpoints showed why
  checkpoint screening is required. GPU1 `model_4422.pt` fixed the hip
  pitch/roll failures but regressed both knees in the full grid. GPU1
  intermediate `model_4250.pt` was close, but failed full-grid
  `right_hip_yaw_joint` and `left_knee_joint`.
- 2026-05-20 The GPU0 seed produced the accepted fixed `1.4 m/s` checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/2026-05-20_22-42-37_009_phasehiprollpitchguard_fast1p4_resume4173_env8192_iter250_gpu0_seed29230/model_4400.pt`.
  Full-grid evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p4_eval_phasehiprollpitchguard_seed29230_model4400_v1p4_full_grid/task029_eval_failure_aggregate.json`.
  Aggregate `pass=true`, clean `lin_vel_error_mean=0.0969`, doubled holdout
  `zero_fall_ratio=0.8398`, and all 12 forced-dead grid cases passed at fixed
  `lin_vel_x=1.4`. The thinnest grid cases were `left_hip_yaw_joint`
  (`zero_fall_ratio=0.793`) and `left_knee_joint`
  (`zero_fall_ratio=0.8633`).
- 2026-05-20 Fixed `1.4 m/s` critical repeated validation passed for the same
  `model_4400.pt` checkpoint. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p4_eval_phasehiprollpitchguard_seed29230_model4400_v1p4_critical_multiseed_s5/task029_dead_motor_knee_multiseed_s5_summary.json`.
  It ran five seeds each for left/right hip-yaw, left/right hip-roll, and
  left/right knee; all 30 cases passed. The limiting repeated cases remained
  `left_hip_yaw_joint` with zero-fall ratios
  `[0.7852, 0.7695, 0.8242, 0.7539, 0.7852]` and `left_knee_joint` with
  `[0.8242, 0.8359, 0.8594, 0.832, 0.8242]`.
- 2026-05-20 Fixed `1.4 m/s` render evidence was generated for the same
  checkpoint. Main render summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/render_phasehiprollpitchguard_model4400_v1p4_left_hip_yaw/task029-render-phasehiprollpitchguard-model4400-v1p4-left-hip-yaw-summary.json`.
  It passed for clean, in-distribution failure, and forced
  `left_hip_yaw_joint` dead. An additional forced `left_knee_joint` dead-only
  render case produced a passing case summary and video under:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/render_phasehiprollpitchguard_model4400_v1p4_left_knee/`.
  Local review copies were also pulled under:
  `D:\guoyubo.9\Documents\New project 2\h200-locomotion-lab\outputs\task029\forward_speed_expansion\`.
- 2026-05-21 Opened the next expansion stage,
  `Unitree-G1-Gripper-Flat-MotorOnly-Failure-PhaseHipRollPitchGuard-Fast1p6`,
  by reusing the accepted HipRollPitchGuard failure sampler and changing only
  the command range to `lin_vel_x=(1.0, 1.6)`. Inspect evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/inspect_hiprollpitchguard_fast1p6/task029_forward_speed_hiprollpitchguard_fast1p6_inspect_summary.json`.
  Inspect passed with `action_dim=31`, `actor_obs=104`, no actor fault-label
  leak, and the same motor-only failure/phase events.
- 2026-05-21 `Fast1p6` 64-env resume smoke from the accepted `Fast1p4`
  `model_4400.pt` completed with no `fell_over` terminations. Smoke log:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p6_smoke/smoke.log`.
- 2026-05-21 Two H200 `Fast1p6` continuation seeds were launched from
  `model_4400.pt` with 8192 envs, 300 iterations, and save interval 50:
  `009_phasehiprollpitchguard_fast1p6_resume4400_env8192_iter300_gpu0_seed29250`
  in tmux `task029_fast1p6_gpu0`, and
  `009_phasehiprollpitchguard_fast1p6_resume4400_env8192_iter300_gpu1_seed29251`
  in tmux `task029_fast1p6_gpu1`.
- 2026-05-21 `Fast1p6` first-pass training found a near miss rather than an
  accepted checkpoint. GPU1 `model_4600.pt` passed the six-joint cheap screen,
  but full-grid failed only `dead_motor_grid_05_left_knee_joint` with
  `zero_fall_ratio=0.4766` against the `0.5` threshold. Full-grid evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p6_eval_phasehiprollpitchguard_seed29251_model4600_v1p6_full_grid/task029_eval_failure_aggregate.json`.
  Clean tracking was healthy (`lin_vel_error_mean=0.1153`) and doubled holdout
  passed (`zero_fall_ratio=0.8242`), so the next route is targeted robustness
  margin, not speed-command tracking.
- 2026-05-21 Opened and inspected
  `PhaseLeftKneeAllCritical-Fast1p6`, which oversamples `left_knee_joint`
  while guarding `right_knee_joint`, both hip-yaw, both hip-roll, and both
  hip-pitch joints. Inspect evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/inspect_leftknee_allcritical_fast1p6/task029_forward_speed_leftknee_allcritical_fast1p6_inspect_summary.json`.
  Inspect passed with the same action/actor-observation contract and
  `lin_vel_x=(1.0, 1.6)`.
- 2026-05-21 `PhaseLeftKneeAllCritical-Fast1p6` 64-env resume smoke from
  GPU1 `model_4600.pt` completed with no `fell_over` terminations. Two short
  8192-env continuation seeds were launched for 150 iterations with save
  interval 25:
  `009_phaseleftkneeallcritical_fast1p6_resume4600_env8192_iter150_gpu0_seed29261`
  and
  `009_phaseleftkneeallcritical_fast1p6_resume4600_env8192_iter150_gpu1_seed29262`.
- 2026-05-21 Accepted fixed `1.6 m/s` with the
  `PhaseLeftKneeAllCritical-Fast1p6` GPU0 checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/2026-05-21_01-33-46_009_phaseleftkneeallcritical_fast1p6_resume4600_env8192_iter150_gpu0_seed29261/model_4700.pt`.
  Full-grid evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p6_eval_phaseleftkneeallcritical_seed29261_model4700_v1p6_full_grid/task029_eval_failure_aggregate.json`.
  Aggregate `pass=true`, clean `lin_vel_error_mean=0.1097`, doubled holdout
  `zero_fall_ratio=0.8125`, and all 12 forced-dead grid cases passed at fixed
  `lin_vel_x=1.6`. The thinnest grid cases were `left_knee_joint`
  (`zero_fall_ratio=0.9453`) and `right_hip_yaw_joint`
  (`zero_fall_ratio=0.9688`).
- 2026-05-21 Fixed `1.6 m/s` critical repeated validation passed for the same
  `model_4700.pt` checkpoint. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p6_eval_phaseleftkneeallcritical_seed29261_model4700_v1p6_critical_multiseed_s5/task029_dead_motor_knee_multiseed_s5_summary.json`.
  It ran five seeds each for left/right hip-yaw, left/right hip-roll, and
  left/right knee; all 30 cases passed. The limiting repeated case was
  `left_knee_joint` with zero-fall ratios
  `[0.9336, 0.9062, 0.9531, 0.9219, 0.9336]`.
- 2026-05-21 Fixed `1.6 m/s` render evidence was generated for the same
  checkpoint. Main render summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/render_phaseleftkneeallcritical_model4700_v1p6_left_knee/task029-render-phaseleftkneeallcritical-model4700-v1p6-left-knee-summary.json`.
  It passed for clean, in-distribution failure, and forced
  `left_knee_joint` dead. An additional forced `right_hip_yaw_joint` dead-only
  render case produced a passing case summary and video under:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/render_phaseleftkneeallcritical_model4700_v1p6_right_hip_yaw/`.
  Local review copies were pulled under:
  `D:\guoyubo.9\Documents\New project 2\h200-locomotion-lab\outputs\task029\forward_speed_expansion\render_fast1p6_model4700\`.

## Review

Status: open. `Fast1p0`, `Fast1p2`, and `Fast1p4` are accepted by H200
full-grid, critical multiseed, and render evidence. `Fast1p6` is also accepted
by H200 full-grid, critical multiseed, and render evidence. The natural next
speed step is `Fast1p8` from the accepted `Fast1p6 model_4700.pt`.
