# 001 Stage2 Left-Knee Partial Train

## Route

Register and train a fixed Stage2 curriculum task with hidden
`left_knee_joint` partial motor failure: `vx=1.3 m/s`, torque scale `0.3`,
onset `1.0 s`, no recovery.

This subtask proves registration, train-pipeline correctness, and checkpoint
writing only. It does not prove hard dead-motor recovery.

## Log

- 2026-08-13 Added local MJLab task
  `Unitree-G1-Gripper-Flat-Task052-TrueTxl-CurriculumStage2-Train`.
  It uses the Task050 `dynamic_motor_failure` actuator-forcerange event with
  template `normal 0.0-1.0s -> left_knee_joint scale 0.3 1.0-999.0s`.
- 2026-08-13 Added wrapper CLI
  `h200_locomotion_lab.tools.task052_fault_curriculum_train`, defaulting to the
  Stage2 task id and preserving the no-recovery-claim boundary.
- 2026-08-13 Smoke continuation from Task049 checkpoint:
  `outputs/task049/bridge_smoke_per_step_mlp_replay/task049_bridge_smoke_per_step_mlp_replay_env512_step24_iter10_mb1_seed4900107/model_9.pt`.
  Result JSON:
  `task052_stage2_left_knee_partial_env512_step24_iter20_mb1_seed5200101.json`.
  Output checkpoint:
  `outputs/task052/curriculum_stage2/task052_stage2_left_knee_partial_env512_step24_iter20_mb1_seed5200101/model_19.pt`.
  Evidence: `train_pipeline_pass=true`, `task052_train_pipeline_pass=true`,
  actor obs shape `[512, 104]`, action shape `[512, 31]`, logprob replay
  parity pass with max logprob and ratio error `0.0`, and
  `task044_fault_aux_updates=40`.
- 2026-08-13 Continued the Stage2 smoke checkpoint for 180 iterations. Result
  JSON:
  `task052_stage2_left_knee_partial_continue_env512_step24_iter180_mb1_seed5200102.json`.
  Output checkpoint:
  `outputs/task052/curriculum_stage2/task052_stage2_left_knee_partial_continue_env512_step24_iter180_mb1_seed5200102/model_179.pt`.
  Evidence: `train_pipeline_pass=true`, `task052_train_pipeline_pass=true`,
  `env_cfg_mode=train`, `task044_fault_aux_updates=360`,
  `task044_fault_aux_last_loss=0.04792369529604912`, and logprob replay parity
  pass with max logprob and ratio error `0.0`.

## Review

Status: passed for the Stage2 train-pipeline gate. Recovery claims are
delegated to subtask 002.
