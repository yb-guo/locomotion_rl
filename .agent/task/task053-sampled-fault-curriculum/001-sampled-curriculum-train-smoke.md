# 001 Sampled Curriculum Train Smoke

## Route

Register and smoke-train the one-shot sampled curriculum task. The curriculum
samples left-knee scale, onset, and command velocity per env at reset and
hardens with `common_step_counter`.

This subtask proves registration, train-pipeline correctness, and checkpoint
writing only. It does not prove hard dead-motor recovery.

## Log

- 2026-08-13 Implemented and verified
  `Unitree-G1-Gripper-Flat-Task053-TrueTxl-SampledCurriculum-Train`.
- 2026-08-13 Checks:
  - compileall passed for the Task053 train wrapper and G1 gripper env cfg;
  - `pytest -q -p no:cacheprovider tests/test_task053_sampled_fault_curriculum_train.py tests/test_task052_fault_curriculum_train.py tests/test_task050_fault_recovery_eval.py`
    passed: `12 passed in 3.30s`;
  - `git diff --check` passed for Task053-touched files.
- 2026-08-13 Smoke train completed on RTX 4090:
  `.agent/task/task053-sampled-fault-curriculum/task053_sampled_curriculum_env512_step24_iter20_mb1_seed5300101.json`.
  Evidence includes `train_pipeline_pass=true`,
  `task053_train_pipeline_pass=true`, `checkpoint_exists=true`, and final
  checkpoint
  `outputs/task053/sampled_curriculum/task053_sampled_curriculum_env512_step24_iter20_mb1_seed5300101/model_19.pt`.
- 2026-08-13 Full sampled-curriculum candidate train completed:
  `.agent/task/task053-sampled-fault-curriculum/task053_sampled_curriculum_env512_step24_iter180_mb1_seed5300102.json`.
  Evidence includes `train_pipeline_pass=true`,
  `task053_train_pipeline_pass=true`, `env_cfg_mode=train`,
  `checkpoint_exists=true`, `final_iteration=179`, and final checkpoint
  `outputs/task053/sampled_curriculum/task053_sampled_curriculum_env512_step24_iter180_mb1_seed5300102/model_179.pt`.

## Review

Status: train smoke and 180-iteration candidate train passed. Hard recovery is
not claimed here.
