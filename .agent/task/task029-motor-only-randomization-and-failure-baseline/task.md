# Task 029: Motor-Only Randomization And Failure Baseline

## Route

Build the next benchmark after task028's fixed-topology G1-like whole-body
gripper environment: a motor-only randomization and persistent motor-failure
baseline that still uses the known-good MJLab + RSL-RL PPO/MLP stack first.

The first acceptance target is deliberately narrow:

- Fixed G1-like topology.
- Fixed link geometry, mass, COM, and inertia.
- Fixed reward, environment, action, and actor observation contract from
  task028.
- Actor does not observe `motor_scale`, `failure_mask`, or explicit fault
  labels.
- Critic may use privileged motor randomization/failure information.
- First policy is the existing MLP PPO baseline.
- First failure mode is episode-start persistent weak/dead leg motors.

Deferred from first acceptance:

- LocoFormer or other long-context policy replacement.
- Sudden mid-episode motor failure.
- Arm, waist, gripper, or upper-body motor failure.
- Locked-joint training.
- Stuck-command training.
- Link mass/COM/inertia randomization.
- Contact friction randomization.
- Encoder noise or observation corruption as a training randomization.

Planned slices:

1. `001-motor-only-contract.md`
   - Define allowed and forbidden randomization fields.
   - Confirm the inherited task028 action/actor-observation contract remains
     unchanged.
   - Define actor/critic information boundaries for motor fault data.

2. `002-motor-primitive-ranges.md`
   - Add motor-side primitive stages one group at a time:
     `kp/kd`, effort/strength scale, damping/friction,
     torque noise/bias, and deadband.
   - Require inspect evidence and short PPO smoke for each primitive.

3. `003-delay-bandwidth-step-response.md`
   - Add action/actuator delay and low-pass/bandwidth behavior.
   - Validate with a step-response harness before PPO training.

4. `004-motor-failure-stage.md`
   - Implement episode-start persistent leg motor failure.
   - Randomly choose `0-2` leg motors per episode.
   - Use weak scale `0.3-0.7` and dead scale `0.0-0.1`.

5. `005-mlp-baseline-train.md`
   - Train the existing MLP PPO baseline on the motor-only/failure stage.
   - Prove the baseline still walks under the first acceptance setting.

6. `006-ood-and-failure-grid-eval.md`
   - Run clean eval, motor-only randomized eval, doubled motor holdout, and
     a per-joint dead-motor grid.
   - Keep locked-joint and stuck-command cases as eval holdouts only.

7. `007-render-and-review.md`
   - Render the accepted checkpoint.
   - Review gait quality and detect reward hacking such as excessive shaking,
     dragging, or upper-body flailing.

8. `008-phase-balanced-knee-failure-ablation.md`
   - Try the minimal fix for the final left-knee-dead failure: reset-time gait
     phase randomization and balanced single-dead left/right knee sampling.
   - Keep a no-phase-randomization control ablation if phase randomization
     regresses clean walking.
   - Keep the same actor/action contract and MLP PPO baseline.

9. `009-forward-speed-expansion.md`
   - Extend the accepted BalancedKnee checkpoint to faster forward commands.
   - Start with fixed-command `1.0 m/s`, then advance toward `1.2 m/s+`.
   - Preserve the task029 motor-only and no actor fault-label contract.

## Minimal Closed Loop

Feedback loop:

1. Inspect the motor-only contract and prove forbidden link/contact/sensor
   randomization is disabled.
2. Add motor primitive stages one at a time and run short H200 PPO smokes.
3. Validate delay/bandwidth behavior with a deterministic step-response
   harness before using it in PPO.
4. Prove the persistent motor-failure sampler with reset statistics and forced
   single-motor traces.
5. Train the existing MLP PPO baseline on the first acceptance distribution.
6. Evaluate the same checkpoint on clean, in-distribution motor-randomized,
   doubled motor holdout, and dead-motor grid scenarios.
7. Render accepted and diagnostic cases from the same checkpoint used in eval.

Pass:

- The actor observation and 31-dim action contract remain compatible with
  task028.
- Training randomization is motor-only under the first acceptance setting.
- Episode-start persistent weak/dead leg motor eval has saved JSON evidence.
- Clean eval confirms the baseline walking behavior was not destroyed.
- Render evidence does not show obvious reward hacking in accepted cases.

Fail:

- Actor receives explicit motor failure labels or motor scales.
- Link mass/COM/inertia, contact friction, sensor corruption, or pushes are
  enabled in the first-pass training stage.
- Delay/bandwidth is accepted without a step-response timing trace.
- Training reward is used as the only evidence without closed-loop eval.
- The task is marked passed before H200 eval and render evidence exist.

Evidence:

- Planned root:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/`.

## Log

- 2026-05-19 Opened after task028 passed fixed-topology G1-like whole-body
  gripper training/eval/render with the existing MLP PPO stack.
- 2026-05-19 User decision: task029 first acceptance is not sudden online
  fault adaptation. It is episode-start persistent weak/dead leg motor
  robustness with the existing MLP baseline.
- 2026-05-19 User decision: actor must not receive explicit failure labels or
  motor scales. Critic may receive privileged motor information for value
  learning and diagnostics.
- 2026-05-19 Subtask 001 passed on H200. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/motor_only_contract/summary.json`.
  Six MotorOnly tasks passed with `action_dim=31`, `actor_obs=104`,
  `critic_obs=119`, actor corruption disabled, and forbidden
  link/contact/sensor/push events absent.
- 2026-05-19 Subtask 002 passed for currently exposed MJLab motor primitives.
  Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/motor_primitives/motor_primitive_smoke_summary.json`.
  Stages `control`, `kp_kd`, `strength`, `damping_friction`, `armature`, and
  `combined` ran 64-env, 2-iteration H200 PPO smokes on `gpu0`, each with
  `model_1.pt`, agent/env YAML, TensorBoard event output, `upload-model=false`,
  and no residual training process.
- 2026-05-19 Subtask 002 explicitly does not cover torque noise, torque bias,
  or deadband because there is no ready MJLab API for those knobs. They remain
  future actuator-wrapper work.
- 2026-05-19 Subtask 003 passed on H200. Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/delay_bandwidth_step_response/summary.json`.
  Step response preserved `action_dim=31` and `actor_obs=104`; baseline raw
  action was unmutated/no-delay, delay-only shifted by `delay_steps=2`,
  low-pass smoothed with `low_pass_alpha=0.35`, and delay+low-pass shifted the
  smoothed target.
- 2026-05-19 Subtask 004 passed for the persistent leg motor-failure stage.
  Evidence root:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/motor_failure_stage/`.
  Stage evidence includes passing inspect, reset/failure diagnostics, forced
  weak/dead trace, and 64-env PPO smoke summaries.
- 2026-05-19 Subtask 005 training completed on H200. Final checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_mlp_baseline_005_train/2026-05-19_20-45-49_005_failure_env8192_iter1200_gpu1_seed29005/model_1199.pt`.
  Training completion alone does not establish accepted robustness.
- 2026-05-19 Subtask 006 completed the final same-checkpoint eval on
  `model_1199.pt`. Final aggregate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/model1199/task029_eval_failure_aggregate.json`.
  Aggregate `pass=false`; the only failing final case is
  `dead_motor_grid_05_left_knee_joint`, with `zero_fall_ratio=0.0` and
  `max_done_count=5`.
- 2026-05-19 Subtask 007 render completed for the final same checkpoint.
  Final render summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/render_review/model1199/task029-render-model1199-summary.json`.
  Summary `pass=true`; videos, midframes, and JSON evidence exist for clean,
  in-distribution failure, and forced left-knee-dead cases.
- 2026-05-19 Terminal state: 001-004 have stage evidence, 005 produced the
  final `model_1199.pt`, 006 completed but failed
  `dead_motor_grid_05_left_knee_joint`, and 007 render completed with
  `pass=true`. Overall task029 is not passed.
- 2026-05-20 Opened subtask 008 after diagnosis showed the left-knee-dead
  failure is stable across repeated seeds while static left/right knee
  actuator, XML, default pose, action-scale, and force-range injection checks
  are symmetric. Subtask 008 will test whether randomizing reset gait phase and
  oversampling single-dead left/right knee episodes removes the learned
  left-phase dependency without exposing fault labels to the actor.
- 2026-05-20 Subtask 008 completed. The phase-randomized ablation fixed the
  dead-motor grid but was rejected because clean tracking narrowly missed the
  threshold (`lin_vel_error_mean=0.3512480556964874` vs `0.35`). The accepted
  fix is the no-phase-randomization balanced-knee-only task:
  `Unitree-G1-Gripper-Flat-MotorOnly-Failure-BalancedKnee`.
- 2026-05-20 Accepted checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/2026-05-20_13-18-04_008_balancedknee_env8192_iter1200_gpu0_seed29109/model_1199.pt`.
  Final aggregate:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/eval/model1199/task029_eval_failure_aggregate.json`.
  Aggregate `pass=true`, `complete_grid=true`, `grid_case_count=12`,
  `grid_case_count_equals_target_count=true`, and
  `clean_motor_failure_stripped=true`.
- 2026-05-20 Accepted final metrics: clean `zero_fall_ratio=1.0`,
  `max_done_count=0`, `lin_vel_error_mean=0.2591520845890045`;
  in-distribution failure `pass=true`; doubled holdout `pass=true`; all 12
  dead-motor grid cases passed. The original failed
  `dead_motor_grid_05_left_knee_joint` now passes with
  `zero_fall_ratio=1.0`, `max_done_count=0`.
- 2026-05-20 Repeated left/right knee-dead validation passed. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/multiseed_model1199_left_right_knee_s10/task029_dead_motor_knee_multiseed_s10_summary.json`.
  It ran 10 seeds for `left_knee_joint` and 10 seeds for
  `right_knee_joint`; both knees were `10/10` pass with all
  `zero_fall_ratio=1.0` and all `max_done_count=0`.
- 2026-05-20 Accepted render evidence passed. Summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/render_model1199/task029-render-balancedknee-model1199-summary.json`.
  Videos exist for clean, in-distribution failure, and left-knee-dead accepted
  cases under:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/balanced_knee_failure/render_model1199/`.
- 2026-05-20 Opened subtask 009 after probing the accepted BalancedKnee
  checkpoint at fixed `0.8 m/s`. It stayed upright but tracked too slowly:
  clean `lin_vel_error_mean=0.5269265174865723`. Subtask 009 will train
  forward-speed variants, starting with `Fast1p0`.
- 2026-05-20 Subtask 009 accepted fixed `1.0 m/s` with the
  PhaseBalancedCritical `Fast1p0` checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/2026-05-20_17-33-37_009_phasebalancedcritical_fast1p0_resume2797b_env8192_iter800_gpu0_seed29153/model_3100.pt`.
  Full eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p0_eval_phasecritical_model3100_v1p0_full_grid/task029_eval_failure_aggregate.json`.
  Aggregate `pass=true`, clean `lin_vel_error_mean=0.10559709370136261`,
  doubled holdout passed, and all 12 forced-dead grid cases passed at fixed
  `lin_vel_x=1.0`.
- 2026-05-20 Subtask 009 render passed for the same `model_3100.pt` checkpoint
  at fixed `1.0 m/s`. Render summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/render_phasecritical_model3100_v1p0_right_hip_yaw/task029-render-phasecritical-model3100-v1p0-summary.json`.
  `Fast1p2` training has started from `model_3100.pt`.
- 2026-05-20 Subtask 009 accepted fixed `1.2 m/s` with the
  PhaseYawKneeQuad checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/2026-05-20_20-46-25_009_phaseyawkneequad_fast1p2_resume3650_env8192_iter75_gpu0_seed29200/model_3675.pt`.
  Full eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p2_eval_phaseyawkneequad_seed29200_model3675_v1p2_full_grid/task029_eval_failure_aggregate.json`.
  Aggregate `pass=true`, `complete_grid=true`, all 12 forced-dead grid cases
  passed at fixed `lin_vel_x=1.2`; the limiting case was forced
  `right_knee_joint` dead with `zero_fall_ratio=0.5234375`.
- 2026-05-20 Subtask 009 repeated critical eval and render also passed for
  `Fast1p2 model_3675.pt`. Critical s5:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p2_eval_phaseyawkneequad_seed29200_model3675_v1p2_critical_multiseed_s5/task029_dead_motor_knee_multiseed_s5_summary.json`.
  Render:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/render_phaseyawkneequad_model3675_v1p2_right_knee/task029-render-phaseyawkneequad-model3675-v1p2-summary.json`.
-  `Fast1p4` training then ran from the accepted `1.2 m/s` checkpoint.
  The first `PhaseYawKneeQuad-Fast1p4` continuation did not close:
  `model_3974.pt` on both seeds failed only the fixed `1.4 m/s`
  `left_knee_joint` forced-dead critical gate, while hip-yaw and right-knee
  guards passed.
- 2026-05-20 A narrower `PhaseLeftKneeGuard-Fast1p4` continuation was opened
  for subtask 009. It keeps the same action/obs/reward/topology contract and
  only reweights training reset failures toward `left_knee_joint` while
  retaining `right_knee_joint`, `left_hip_yaw_joint`, and
  `right_hip_yaw_joint` as guards. Inspect passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/inspect_leftkneeguard/task029_forward_speed_leftkneeguard_inspect_summary.json`.
  Its GPU1 `model_4173.pt` fixed the left-knee critical gate, but full-grid
  evaluation then failed hip pitch/roll cases at fixed `1.4 m/s`.
- 2026-05-20 Subtask 009 moved to `PhaseHipRollPitchGuard-Fast1p4`, which
  keeps the same contract and adds right/left hip-roll plus hip-pitch guard
  sampling while retaining knee/yaw guards. Inspect passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/inspect_hiprollpitchguard/task029_forward_speed_hiprollpitchguard_inspect_summary.json`.
  Two H200 continuation seeds are running from the best current left-knee
  guard checkpoint.
- 2026-05-20 Subtask 009 accepted fixed `1.4 m/s` with the
  `PhaseHipRollPitchGuard-Fast1p4` GPU0 checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/2026-05-20_22-42-37_009_phasehiprollpitchguard_fast1p4_resume4173_env8192_iter250_gpu0_seed29230/model_4400.pt`.
  Full-grid evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p4_eval_phasehiprollpitchguard_seed29230_model4400_v1p4_full_grid/task029_eval_failure_aggregate.json`.
  Aggregate `pass=true`, clean `lin_vel_error_mean=0.0969`, doubled holdout
  `zero_fall_ratio=0.8398`, and all 12 forced-dead grid cases passed at fixed
  `lin_vel_x=1.4`.
- 2026-05-20 Fixed `1.4 m/s` critical repeated validation and render evidence
  also passed for `model_4400.pt`. Critical s5 evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p4_eval_phasehiprollpitchguard_seed29230_model4400_v1p4_critical_multiseed_s5/task029_dead_motor_knee_multiseed_s5_summary.json`.
  Render summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/render_phasehiprollpitchguard_model4400_v1p4_left_hip_yaw/task029-render-phasehiprollpitchguard-model4400-v1p4-left-hip-yaw-summary.json`.
  `Fast1p0`, `Fast1p2`, and `Fast1p4` now have closed-loop eval and render
  evidence; subtask 009 remains open only because the requested speed expansion
  is continuing above `1.4 m/s`.
- 2026-05-21 Subtask 009 opened `Fast1p6` by reusing the accepted
  HipRollPitchGuard failure sampler and changing only `lin_vel_x` to
  `(1.0, 1.6)`. Inspect passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/inspect_hiprollpitchguard_fast1p6/task029_forward_speed_hiprollpitchguard_fast1p6_inspect_summary.json`.
  A 64-env resume smoke from `Fast1p4 model_4400.pt` completed, and two
  8192-env H200 continuation seeds are running:
  `009_phasehiprollpitchguard_fast1p6_resume4400_env8192_iter300_gpu0_seed29250`
  and
  `009_phasehiprollpitchguard_fast1p6_resume4400_env8192_iter300_gpu1_seed29251`.
- 2026-05-21 The first `Fast1p6` training pass produced a near miss: GPU1
  `model_4600.pt` failed only `dead_motor_grid_05_left_knee_joint` in the
  full grid (`zero_fall_ratio=0.4766` vs threshold `0.5`) while clean tracking
  remained healthy (`lin_vel_error_mean=0.1153`). Evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p6_eval_phasehiprollpitchguard_seed29251_model4600_v1p6_full_grid/task029_eval_failure_aggregate.json`.
- 2026-05-21 Subtask 009 opened
  `PhaseLeftKneeAllCritical-Fast1p6`, preserving the action/obs/reward/topology
  contract but oversampling left-knee dead episodes while guarding right knee,
  hip-yaw, hip-roll, and hip-pitch cases. Inspect passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/inspect_leftknee_allcritical_fast1p6/task029_forward_speed_leftknee_allcritical_fast1p6_inspect_summary.json`.
  A 64-env resume smoke from `model_4600.pt` completed, and two short 8192-env
  H200 continuation seeds are running for 150 iterations:
  `009_phaseleftkneeallcritical_fast1p6_resume4600_env8192_iter150_gpu0_seed29261`
  and
  `009_phaseleftkneeallcritical_fast1p6_resume4600_env8192_iter150_gpu1_seed29262`.
- 2026-05-21 Subtask 009 accepted fixed `1.6 m/s` with the
  `PhaseLeftKneeAllCritical-Fast1p6` GPU0 checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/external/unitree_rl_mjlab/logs/rsl_rl/g1_gripper_velocity_task029_balanced_knee_008_train/2026-05-21_01-33-46_009_phaseleftkneeallcritical_fast1p6_resume4600_env8192_iter150_gpu0_seed29261/model_4700.pt`.
  Full-grid evidence:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p6_eval_phaseleftkneeallcritical_seed29261_model4700_v1p6_full_grid/task029_eval_failure_aggregate.json`.
  Aggregate `pass=true`, clean `lin_vel_error_mean=0.1097`, doubled holdout
  `zero_fall_ratio=0.8125`, and all 12 forced-dead grid cases passed at fixed
  `lin_vel_x=1.6`.
- 2026-05-21 Fixed `1.6 m/s` critical repeated validation and render evidence
  also passed for `model_4700.pt`. Critical s5:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/fast1p6_eval_phaseleftkneeallcritical_seed29261_model4700_v1p6_critical_multiseed_s5/task029_dead_motor_knee_multiseed_s5_summary.json`.
  Render summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/forward_speed_expansion/render_phaseleftkneeallcritical_model4700_v1p6_left_knee/task029-render-phaseleftkneeallcritical-model4700-v1p6-left-knee-summary.json`.
  `Fast1p0`, `Fast1p2`, `Fast1p4`, and `Fast1p6` now have closed-loop eval
  and render evidence.

## Review

Status: passed for the original motor-failure baseline; extended forward-speed
work continues in subtask 009.

This task is designed as a diagnostic bridge between task028's environment
learnability proof and any later LocoFormer-style long-context policy work. If
the MLP baseline cannot handle persistent episode-start leg motor failures, the
next action is to diagnose motor randomization ranges, actuator wrappers,
reward stability, or failure sampling. It is not yet evidence that a larger
policy is required.

The first pass is intentionally motor-only. Any training result that mixes in
link mass/COM/inertia, contact friction, sensor corruption, or explicit actor
fault labels does not satisfy this task's core acceptance criteria.

Subtasks 001, 002, 003, and 004 have H200 stage evidence. Subtask 005 completed
the first full training run, and subtask 006/007 exposed a real left-knee-dead
robustness failure in that checkpoint. Subtask 008 then accepted a minimal
training-distribution fix: balanced single-dead left/right knee oversampling
without reset phase randomization.

Overall task029 is passed by the subtask 008 accepted checkpoint. It preserves
the fixed-topology, motor-only, no actor fault-label contract; passes the clean,
motor-randomized, doubled holdout, and full dead-motor-grid eval gate; passes
repeated left/right knee-dead validation; and has accepted render evidence.
The later speed-expansion subtask has accepted fixed-command `1.0`, `1.2`,
`1.4`, and `1.6 m/s` checkpoints with the same no actor fault-label contract.
The natural next speed step is `Fast1p8` from the accepted `Fast1p6
model_4700.pt`.
