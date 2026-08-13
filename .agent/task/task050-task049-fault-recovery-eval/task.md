# Task 050: Task049 Fault Recovery Eval

## Route

Evaluate whether the accepted Task049 short-continuation checkpoint handles a
hidden damaged joint. Do not infer this from clean-gait evidence.

Checkpoint under test:

- `outputs/task049/bridge_smoke_per_step_mlp_replay/task049_bridge_smoke_per_step_mlp_replay_env512_step24_iter10_mb1_seed4900107/model_9.pt`

Routes:

1. `001-continuous-left-knee-dead-eval.md`
   - Run Task044 continuous no-inner-reset hidden left-knee dead eval.
   - This answers deployment-style "can it keep going without a physical reset?"

2. `002-retry-left-knee-dead-eval.md`
   - Run Task037/Task046 multi-trial retry eval with memory preserved across
     inner resets.
   - This answers "after a fall/reset, does retry improve?"

## Acceptance Criteria

- Continuous eval JSON exists and records post-fault quality gate metrics.
- Retry eval JSON exists and records trial0/trial1/final-trial metrics.
- The review clearly separates:
  - clean gait;
  - continuous damaged-joint recovery;
  - retry-after-fall recovery.
- No claim is made for all damaged joints from a single left-knee eval.
- No external checkpoint, dataset, asset, or upstream repo is downloaded.

## Log

- 2026-08-12 Opened after the user asked whether context can recover under
  damaged joints. Current evidence before this task is clean-only for Task049.
- 2026-08-12 First continuous attempt with the historical Task044 task id wrote
  `outputs/task050/continuous_left_knee_dead/task049_model9_left_knee_dead_continuous_seed5000101.json`
  but failed before runtime eval with
  `KeyError('Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenPoseTight1p6')`.
- 2026-08-12 Second continuous attempt using
  `Unitree-G1-Gripper-Flat-Task048-TrueTxl-CleanBins-Eval` wrote
  `outputs/task050/continuous_left_knee_dead/task049_model9_task048base_left_knee_dead_continuous_seed5000101.json`
  but failed before runtime eval with
  `RuntimeError('dynamic_motor_failure event is absent')`.
- 2026-08-12 Added local Task050 dynamic-fault eval registration:
  `Unitree-G1-Gripper-Flat-Task050-TrueTxl-DynamicFault-Eval`.
  The event preserves the Task048 actor observation contract and applies hidden
  per-step `actuator_forcerange` scaling for the selected damaged joint.
- 2026-08-12 Verification before full eval:
  `python -m compileall external/unitree_rl_mjlab/src/tasks/velocity/config/g1_gripper/env_cfgs.py external/unitree_rl_mjlab/src/tasks/velocity/config/g1_gripper/__init__.py tests/test_task050_fault_recovery_eval.py`;
  `pytest -q tests/test_task050_fault_recovery_eval.py` -> `2 passed`;
  CPU smoke showed `left_knee_joint` ctrl id `16` force range changes from
  `[-139, 139]` to `[0, 0]` after the step event and resets to default on env
  reset.
- 2026-08-12 Continuous no-reset eval completed:
  `outputs/task050/continuous_left_knee_dead/task049_model9_task050_left_knee_dead_continuous_seed5000101.json`.
  Result: `pipeline_pass=false`, `quality_gate_pass=false`,
  `physical_continuity_pass=false`, `physical_reset_events=512`,
  `physical_fall_events=512`, `post_fault_window.fall_ratio=1.0`,
  `post_fault_window.lin_vel_error.mean=0.6310434341430664`,
  `post_fault_window.gravity_xy.max=0.9543123841285706`, and
  `post_fault_window.root_z.min=0.27702611684799194`.
- 2026-08-12 Retry-after-fall eval completed:
  `outputs/task050/retry_left_knee_dead/task049_model9_task050_left_knee_dead_retry_seed5000201.json`.
  Result: `final_trial_pass=false`; trial fall ratios were all `1.0`
  (`trial_0`, `trial_1`, and `final_trial`), final-trial
  `lin_vel_error.mean=1.6589775085449219`,
  `yaw_vel_error.mean=1.059837818145752`,
  `gravity_xy.max=0.9453970789909363`, and
  `root_z.min=0.23644568026065826`.

## Review

Status: closed with negative fault-recovery evidence.

Task049 remains a clean-gait checkpoint only. Its clean matrix passed at
`0.4/1.2/2.0 m/s`, but the left-knee dead-motor runtime eval did not recover.
The True-TXL context was active in both fault evals, yet:

- continuous damaged-joint recovery is not demonstrated because the no-reset
  eval had 512 physical fall/reset events and failed the post-fault gate;
- retry-after-fall recovery is not demonstrated because the final trial still
  had `fall_ratio=1.0` and failed the promotion gate;
- this is only a hidden `left_knee_joint` dead-motor eval, not an all-joint
  damaged-joint claim.
