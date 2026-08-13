# 001 Left-Knee Hidden-Fault Train Smoke

## Route

Register and smoke-train a Task051 True-TXL continuation task with hidden
`left_knee_joint` dead motor at `vx=1.6 m/s`. This subtask proves the training
pipeline and checkpoint writing only; it does not prove damaged-joint recovery.

## Log

- 2026-08-12 Added local MJLab task
  `Unitree-G1-Gripper-Flat-Task051-TrueTxl-LeftKneeDead-Train`.
  It uses the Task050 `dynamic_motor_failure` actuator-forcerange event with
  template `normal 0.0-0.5s -> left_knee_joint dead 0.5-999.0s`.
- 2026-08-12 Added wrapper CLI
  `h200_locomotion_lab.tools.task051_fault_specialized_train` so the base
  Task041 training preflight can run against the Task051 task id without
  weakening Task041's own task boundary.
- 2026-08-12 Smoke continuation from Task049 checkpoint:
  `outputs/task049/bridge_smoke_per_step_mlp_replay/task049_bridge_smoke_per_step_mlp_replay_env512_step24_iter10_mb1_seed4900107/model_9.pt`.
  Result JSON:
  `task051_left_knee_fault_env512_step24_iter10_mb1_seed5100101.json`.
  Output checkpoint:
  `outputs/task051/left_knee_fault_train/task051_left_knee_fault_env512_step24_iter10_mb1_seed5100101/model_9.pt`.
  Evidence: `train_pipeline_pass=true`, `task051_train_pipeline_pass=true`,
  `env_cfg_mode=train`, runner `Task044TrueTxlMemoryK160ClearHistoryRunner`,
  algorithm `Task040SequenceAwareTrueTxlPPO`, actor
  `Task038TrueTxlMemoryModel`, actor obs shape `[512, 104]`, action shape
  `[512, 31]`, logprob replay parity pass with max logprob and ratio error
  `0.0`, and `task044_fault_aux_updates=20`.
- 2026-08-12 Continued the Task051 smoke checkpoint for 90 more iterations.
  Result JSON:
  `task051_left_knee_fault_continue_env512_step24_iter90_mb1_seed5100102.json`.
  Output checkpoint:
  `outputs/task051/left_knee_fault_train/task051_left_knee_fault_continue_env512_step24_iter90_mb1_seed5100102/model_89.pt`.
  Evidence: `train_pipeline_pass=true`, `task051_train_pipeline_pass=true`,
  `env_cfg_mode=train`, `task044_fault_aux_updates=180`,
  `task044_fault_aux_last_loss=0.28173401951789856`, logprob replay parity
  pass with max logprob and ratio error `0.0`. Training debug recorded
  `sequence_update_reset_events=16122`, so this is a runnable training
  checkpoint, not evidence of stable damaged-joint locomotion.

## Review

Status: passed for the train-pipeline gate. Recovery remains delegated to
subtask 002 and is not claimed here.
