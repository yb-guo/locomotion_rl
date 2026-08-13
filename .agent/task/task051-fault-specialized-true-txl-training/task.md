# Task 051: Fault-Specialized True-TXL Training

## Route

Train a fault-specialized True-TXL policy from the accepted Task049 clean
checkpoint. The first target is deliberately narrow and eval-aligned:

- fixed command `vx=1.6 m/s`;
- hidden `left_knee_joint` dead motor;
- dynamic onset at `0.5 s`, no recovery during the rollout;
- actor observation keeps the existing hidden-fault boundary: no explicit
  fault identity, scale, onset, or recovery label is visible to the actor;
- privileged fault labels may be used only for auxiliary memory-latent training
  diagnostics.

Source checkpoint:

- `outputs/task049/bridge_smoke_per_step_mlp_replay/task049_bridge_smoke_per_step_mlp_replay_env512_step24_iter10_mb1_seed4900107/model_9.pt`

## Planned Slices

1. `001-left-knee-hidden-fault-train-smoke.md`
   - Register a local Task051 training task using the Task050 hidden
     `dynamic_motor_failure` implementation.
   - Run a short PPO continuation from the Task049 checkpoint.
   - This is only a training-pipeline gate.

2. `002-task050-recovery-eval.md`
   - Evaluate the trained checkpoint with the Task050 continuous no-reset eval.
   - Evaluate the same checkpoint with the Task050 retry-after-fall eval.
   - Accept recovery only if both JSON gates pass.

## Acceptance Criteria

- Task051 train task registration exists and uses
  `Task044TrueTxlMemoryK160ClearHistoryRunner` for training.
- Training JSON exists, loads the Task049 checkpoint, records
  `env_cfg_mode=train`, and passes strict sequence replay parity.
- Actor-visible observation does not include explicit fault labels.
- Continuous Task050 eval JSON exists and passes physical continuity plus
  post-fault quality gates.
- Retry Task050 eval JSON exists and passes the final-trial promotion gate.
- No all-joint damaged-joint claim is made from the first left-knee-only target.
- No external checkpoint, dataset, asset, or upstream repository is downloaded.

## Log

- 2026-08-12 Opened at user request to train on hidden dynamic motor failure
  instead of testing the clean-only Task049 checkpoint out of distribution.
  Scope is first narrowed to hidden `left_knee_joint` dead-motor recovery at
  `1.6 m/s`, because Task050 already showed this case fails for Task049.
- 2026-08-12 Added Task051 MJLab train registration
  `Unitree-G1-Gripper-Flat-Task051-TrueTxl-LeftKneeDead-Train`.
  Actor observation remains 104D and does not expose explicit fault identity,
  scale, onset, or recovery labels. Privileged `task044_fault_label` is present
  only for the auxiliary memory-latent diagnostic loss.
- 2026-08-12 Ran fault-specialized continuation from the Task049 checkpoint for
  10 iterations, then continued the Task051 checkpoint for 90 more iterations.
  Latest candidate:
  `outputs/task051/left_knee_fault_train/task051_left_knee_fault_continue_env512_step24_iter90_mb1_seed5100102/model_89.pt`.
  Training pipeline JSON:
  `task051_left_knee_fault_continue_env512_step24_iter90_mb1_seed5100102.json`.
  It reports `train_pipeline_pass=true`, `task051_train_pipeline_pass=true`,
  runner `Task044TrueTxlMemoryK160ClearHistoryRunner`, algorithm
  `Task040SequenceAwareTrueTxlPPO`, actor `Task038TrueTxlMemoryModel`,
  strict logprob replay parity pass with max logprob and ratio error `0.0`,
  and `task044_fault_aux_updates=180`.
- 2026-08-12 Evaluated the 90-iteration candidate with Task050 continuous and
  retry eval. Both gates failed. Continuous eval:
  `outputs/task051/eval_left_knee_fault_train/task051_model89_task050_left_knee_dead_continuous_seed5100202.json`
  reports `pipeline_pass=false`, `physical_continuity_pass=false`,
  `physical_reset_events=512`, `physical_fall_events=512`, post-fault
  `fall_ratio=1.0`, mean linear velocity error `0.7162381410598755`,
  max `gravity_xy=0.9545283317565918`, and min `root_z=0.30517569184303284`.
  Retry eval:
  `outputs/task051/eval_left_knee_fault_train/task051_model89_task050_left_knee_dead_retry_actorcfg_seed5100302.json`
  reports `pass=false`, `final_trial_pass=false`, aggregate
  `fall_ratio=1.0`, aggregate `fall_count=1024`, and final-trial
  `fall_count=256`.

## Review

Status: active. The train-pipeline slice is verified, but the current
fault-specialized checkpoint is rejected by the Task050 recovery gates. No
left-knee recovery or all-joint damaged-joint robustness claim is supported.
