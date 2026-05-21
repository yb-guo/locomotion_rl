# 008: Phase-Balanced Knee-Failure Ablation

## Route

Diagnose and try the smallest training-side fix for the final task029 failure:
`model_1199.pt` consistently falls when `left_knee_joint` is forced dead while
the mirrored `right_knee_joint` case survives.

This ablation keeps the task029 first-pass contract:

- Fixed G1-like topology.
- Fixed link geometry, mass, COM, and inertia.
- Fixed reward, action, and actor observation dimensions.
- Actor still receives no explicit motor scale, failure mask, or fault label.
- Policy remains the existing MLP PPO baseline.

The only changes are reset-time training distribution changes:

1. Randomize gait phase by offsetting `episode_length_buf` at reset while
   preserving the same `phase` observation definition and `foot_gait` reward.
2. Oversample single-dead knee episodes by forcing a fraction of resets to
   exactly one dead knee, sampled between `left_knee_joint` and
   `right_knee_joint`.
3. Leave the remaining resets on the original `0-2` weak/dead leg-motor
   sampler.

## Minimal Closed Loop

Feedback loop:

1. Register a separate MJLab task so the original task029 baseline remains
   reproducible.
2. Inspect the registered task and confirm the actor/action contract is still
   compatible with task028/task029.
3. Run a short PPO smoke before long training.
4. Train the same MLP PPO stack.
5. Re-run the existing same-checkpoint eval gate plus the multi-seed left/right
   knee-dead diagnostic.

Pass:

- `action_dim=31`, `actor_obs=104`, and no actor fault-label leak.
- Reset logs show nonzero phase offsets and left/right knee single-dead
  oversampling.
- Clean eval remains stable.
- Both `left_knee_joint` and `right_knee_joint` forced-dead cases pass the
  dead-motor grid threshold across repeated seeds.

Fail:

- Actor receives explicit failure labels, motor scales, or sampled phase offset.
- Clean walking regresses below the task029 threshold.
- Left/right knee failure remains asymmetric after repeated seeds.
- Training only improves the grid by overfitting to one knee and breaking the
  mirrored knee.

Evidence:

- Planned H200 output root:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/phase_balanced_knee_failure/`.

## Log

- 2026-05-20 Opened after diagnosis showed `model_1199.pt` failure is stable
  across 10 seeds for left-knee dead and not reproduced for right-knee dead.
  Static checks found symmetric knee XML limits, default pose rules, action
  scale rules, and dead-motor force-range injection. The likely failure mode is
  a learned fixed-phase gait asymmetry rather than a bad left-knee motor
  mapping.
- 2026-05-20 Added local H200 patcher and launcher artifacts:
  `task029_create_phase_balanced_failure_stage.py` and
  `task029_start_phase_balanced_training.py`. Local `--help` passed for both;
  AST syntax checks passed. `py_compile` was not used as evidence because the
  existing ignored `__pycache__` path denies pyc replacement on Windows.
- 2026-05-20 Applied the patcher on H200. New registered task:
  `Unitree-G1-Gripper-Flat-MotorOnly-Failure-PhaseBalanced`.
- 2026-05-20 H200 contract probe passed. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/phase_balanced_knee_failure/status_probe/task029_phase_balanced_status_summary.json`.
  The probe confirmed `action_dim=31`, actor observation dim `104`, critic dim
  `119`, no actor motor/failure/fault/scale leak, reset phase offsets spanning
  `0..29` control steps for a `0.6s` period, and oversampled left/right knee
  reset hits.
- 2026-05-20 H200 64-env, 2-iteration PPO smoke completed on GPU1. Status:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/phase_balanced_knee_failure/smoke/task029_phase_balanced_status_summary.json`.
  Smoke produced `model_1.pt`, agent/env yaml, and TensorBoard events under:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_phase_balanced_008_train/2026-05-20_12-14-25_008_phasebalanced_smoke_env64_iter2_gpu1_seed29108/`.
- 2026-05-20 Started formal 8192-env, 1200-iteration PPO training on GPU1
  with seed `29108`. Launch/status root:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/phase_balanced_knee_failure/train/`.
  Training log dir:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_phase_balanced_008_train/2026-05-20_12-15-52_008_phasebalanced_env8192_iter1200_gpu1_seed29108/`.
  Early status at iteration `27/1200` showed the process running as PID
  `632956`, TensorBoard/yaml present, `model_0.pt` saved, and throughput around
  `76k-78k` samples/s.
- 2026-05-20 Ran a mid-training same-checkpoint eval on `model_100.pt` while
  the formal training continued on GPU1. Aggregate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/phase_balanced_knee_failure/eval/model100/task029_eval_failure_aggregate.json`.
  Aggregate `pass=false`, as expected for an early checkpoint: clean tracking
  failed with `lin_vel_error_mean=0.5032078623771667`. The targeted diagnostic
  improved: `dead_motor_grid_05_left_knee_joint` passed with
  `zero_fall_ratio=1.0`, `max_done_count=0`; `dead_motor_grid_07_right_knee_joint`
  also passed with `zero_fall_ratio=1.0`, `max_done_count=0`. Remaining early
  hard failures included left/right ankle pitch and left hip pitch.
- 2026-05-20 Ran a second mid-training eval on `model_300.pt`. Aggregate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/phase_balanced_knee_failure/eval/model300/task029_eval_failure_aggregate.json`.
  Aggregate remained `pass=false`: clean tracking still failed with
  `lin_vel_error_mean=0.5199576616287231`. The targeted knee cases still
  passed: left knee dead `zero_fall_ratio=1.0`, `max_done_count=0`; right knee
  dead `zero_fall_ratio=1.0`, `max_done_count=0`. The remaining failed grid
  case was `dead_motor_grid_03_right_hip_yaw_joint`.
- 2026-05-20 Ran a third mid-training eval on `model_600.pt`. Aggregate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/phase_balanced_knee_failure/eval/model600/task029_eval_failure_aggregate.json`.
  Aggregate remained `pass=false`; clean tracking degraded to
  `lin_vel_error_mean=0.590717077255249`. Both knee-dead cases still passed.
  This suggests reset phase randomization may be making the MLP ignore or
  mistrust phase and settle into a slow/standing gait.
- 2026-05-20 Added a parallel control ablation without reset phase
  randomization:
  `Unitree-G1-Gripper-Flat-MotorOnly-Failure-BalancedKnee`. It keeps the
  balanced single-dead left/right knee oversampling but leaves the original
  phase clock intact.
- 2026-05-20 H200 balanced-knee-only contract probe passed. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/status_probe/task029_phase_balanced_status_summary.json`.
  The probe confirmed `action_dim=31`, actor observation dim `104`, critic dim
  `119`, no actor fault-label leak, no `phase_randomization` event, and
  amplified left/right knee reset hits.
- 2026-05-20 H200 balanced-knee-only 64-env, 2-iteration PPO smoke passed.
  Status:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/smoke/task029_phase_balanced_status_summary.json`.
  Smoke produced `model_1.pt`, agent/env yaml, and TensorBoard events under:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/2026-05-20_13-16-40_008_balancedknee_smoke_env64_iter2_gpu0_seed29109/`.
- 2026-05-20 Started formal balanced-knee-only 8192-env, 1200-iteration PPO
  training on GPU0 with seed `29109`. Launch/status root:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/train/`.
  Training log dir will be under:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/`.
- 2026-05-20 The phase-randomized final checkpoint completed, but it is not
  the accepted solution. Correctly re-run final aggregate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/phase_balanced_knee_failure/eval/model1199_rerun/task029_eval_failure_aggregate.json`.
  It produced a complete 12-case grid and fixed all dead-motor grid cases,
  including left/right knee dead, but aggregate `pass=false` because clean
  tracking narrowly missed the threshold:
  `clean_forward_0p5 lin_vel_error_mean=0.3512480556964874` against the
  `0.35` clean threshold. Earlier `model1199/` output is invalid because the
  first run used the wrong cwd and all cases failed with `ModuleNotFoundError:
  No module named 'src'`.
- 2026-05-20 Balanced-knee-only mid-training eval showed the intended trend.
  `model_400.pt` and `model_500.pt` were still failing clean speed and
  hip-yaw dead-motor cases, but both left/right knee-dead cases passed.
  `model_800.pt` and `model_900.pt` passed the full dead-motor grid; only
  clean speed remained above threshold. Aggregate paths:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/eval/model400/task029_eval_failure_aggregate.json`,
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/eval/model500/task029_eval_failure_aggregate.json`,
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/eval/model800/task029_eval_failure_aggregate.json`,
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/eval/model900/task029_eval_failure_aggregate.json`.
- 2026-05-20 Balanced-knee-only formal training completed. Accepted
  checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/2026-05-20_13-18-04_008_balancedknee_env8192_iter1200_gpu0_seed29109/model_1199.pt`.
  Final same-checkpoint aggregate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/eval/model1199/task029_eval_failure_aggregate.json`.
  Aggregate `pass=true`, `complete_grid=true`, `grid_case_count=12`,
  `grid_case_count_equals_target_count=true`, and
  `clean_motor_failure_stripped=true`.
- 2026-05-20 Final accepted metrics:
  clean `zero_fall_ratio=1.0`, `max_done_count=0`,
  `lin_vel_error_mean=0.2591520845890045`,
  `yaw_vel_error_mean=0.06726361811161041`;
  motor-primitives `pass=true`;
  in-distribution failure `pass=true`, `zero_fall_ratio=0.9921875`;
  doubled holdout `pass=true`, `zero_fall_ratio=1.0`,
  `lin_vel_error_mean=0.3305389881134033`; all 12 dead-motor grid cases
  passed.
- 2026-05-20 Final knee-dead metrics:
  `dead_motor_grid_05_left_knee_joint` passed with
  `zero_fall_ratio=1.0`, `max_done_count=0`,
  `lin_vel_error_mean=0.46166229248046875`,
  `yaw_vel_error_mean=0.09937524795532227`;
  `dead_motor_grid_07_right_knee_joint` passed with
  `zero_fall_ratio=1.0`, `max_done_count=0`,
  `lin_vel_error_mean=0.4480644464492798`,
  `yaw_vel_error_mean=0.07683877646923065`.
- 2026-05-20 Added and ran `task029_dead_motor_knee_multiseed.py` for repeated
  forced-dead knee eval. Local `--help` and AST parse passed. H200 summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/multiseed_model1199_left_right_knee_s10/task029_dead_motor_knee_multiseed_s10_summary.json`.
  It ran 20 cases total: 10 seeds for `left_knee_joint` and 10 seeds for
  `right_knee_joint`, each with 256 envs and 1000 control steps. Summary
  `pass=true`; both knees were `10/10` pass with all `zero_fall_ratio=1.0`
  and all `max_done_count=0`.
- 2026-05-20 Rendered the accepted checkpoint for clean, in-distribution
  failure, and `left_knee_joint` dead cases. Render summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/render_model1199/task029-render-balancedknee-model1199-summary.json`.
  Summary `pass=true`. Videos:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/render_model1199/task029-render-balancedknee-model1199-clean_forward_0p5.mp4`,
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/render_model1199/task029-render-balancedknee-model1199-failure_indistribution_forward_0p5.mp4`,
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/render_model1199/task029-render-balancedknee-model1199-dead_motor_left_knee_joint.mp4`.
- 2026-05-20 Local verification passed after setting `PYTHONPATH=src`:
  `python -m pytest` reported `329 passed, 17 skipped`; `python -m
  h200_locomotion_lab.tools.inspect_agent` completed and printed the expected
  `sonic_adapter` and `locoformer_min` inventory. Running the same commands
  without `PYTHONPATH=src` failed during import collection with
  `ModuleNotFoundError: No module named 'h200_locomotion_lab'`, so those
  failures are environment setup, not task029 regressions.

## Review

Status: passed.

The accepted fix is balanced single-dead knee oversampling without reset phase
randomization. The phase-randomized task fixed the dead-motor grid but
introduced a clean tracking boundary failure, so it should not become the
default curriculum as implemented here.

The balanced-knee-only task preserves the actor/action contract and the
existing MLP PPO baseline, fixes the stable left-knee-dead failure, keeps the
mirrored right-knee case passing, passes the full same-checkpoint eval gate,
passes 10-seed repeated left/right knee-dead diagnostics, and has render
evidence for the accepted checkpoint.
