# 001: Existing Checkpoint Retry Eval

## Route

Start with the existing Task037/Task044 multi-trial eval machinery before
adding a new tool. The first question is empirical:

- current best checkpoint;
- hidden left-knee dead fault;
- multiple attempts with memory preserved across inner resets;
- promote only the final trial metrics.

If this existing path cannot represent the intended 2.0 s hidden-onset retry
contract, document the mismatch and open subtask 002 for a dedicated CLI.

## Acceptance

- H200 eval JSON exists for the current best checkpoint.
- The command records trial 0, trial 1, and final-trial metrics.
- The log states whether final-trial performance improves relative to early
  trials.
- The old continuous gate is not marked passed from this result.

## Log

- 2026-06-02 Opened. Current best checkpoint to test:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/hidden_fault_train/logs_long_survival_all_env2048_iter40_lr5e6_seed4520402/model_39.pt`.
- 2026-06-02 Initial H200 run failed before rollout because the slim conda env
  lacked optional `IPython`, and then because the eval CLI constructed a
  different actor shape from the checkpoint. Fixed the CLI compatibility by
  adding local optional dependency stubs and true-TXL actor cfg overrides.
- 2026-06-02 Local validation:
  `python -m pytest -q -p no:cacheprovider tests\test_task037_mjlab_smoke_scripts.py --tb=short --basetemp '.test_tmp_task046_retry_eval_local4'`
  passed `11 passed, 1 skipped`.
- 2026-06-02 Local validation:
  `python -m h200_locomotion_lab.tools.inspect_agent` passed.
- 2026-06-02 H200 validation:
  `python -m pytest -q -p no:cacheprovider tests/test_task037_mjlab_smoke_scripts.py --tb=short --basetemp /tmp/task046_retry_eval_pytest3`
  passed `12 passed`.
- 2026-06-02 H200 retry eval:
  - checkpoint:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/hidden_fault_train/logs_long_survival_all_env2048_iter40_lr5e6_seed4520402/model_39.pt`
  - output:
    `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_eval/current_best_pose_tight_left_knee_onset0p5_retry_seed4620103.json`
  - args: task
    `Unitree-G1-Gripper-Flat-Task044-TrueTxlHiddenFaultTrain-PersistentHiddenPoseTight1p6`,
    `num_envs=256`, `steps=300`, `trial_length_s=2.0`, `lin_vel_x=1.6`,
    `dynamic_dead_joint=left_knee_joint`, `dynamic_onset_s=0.5`,
    `dynamic_recovery_s=999.0`, `final_window_s=1.0`,
    `memory_latent_dim=32`, `base_obs_passthrough=true`,
    `adaptation_warmstart=true`, `action_dim=31`,
    `adaptation_hidden_dim=128`,
    `base_obs_passthrough_scale=0.5`,
    `adaptation_warmstart_scale=0.5`.
  - result: `pass=true`, `final_trial_pass=true`.
  - trial0: fall_count `4`, fall_ratio `0.0155`, lin_vel_error_mean `0.6077`,
    gravity_xy_max `0.7109`, root_z_min `0.5763`.
  - trial1: fall_count `3`, fall_ratio `0.0117`, lin_vel_error_mean `0.6070`,
    gravity_xy_max `0.7046`, root_z_min `0.5759`.
  - final trial: fall_count `2`, fall_ratio `0.0078`,
    lin_vel_error_mean `0.6069`, gravity_xy_max `0.6791`,
    root_z_min `0.5934`.
  - final tail window: fall_count `2`, fall_ratio `0.0078`,
    lin_vel_error_mean `0.2780`.
  - TXL debug sampled attended previous memory lengths as `[64, 64]`,
    confirming the eval path did not run stateless.

## Review

Status: evidence complete for subtask 001.

The existing Task037 multi-trial eval can represent the first retry-after-fall
question for the current checkpoint. The observed trend is modest but positive:
falls decrease across attempts from `4 -> 3 -> 2`, and the promoted final trial
passes the retry contract.

This does not pass or relax the old Task044/Task045 continuous no-physical-reset
gate. It only shows that retry-after-fall with retained memory is a plausible
separate eval/training contract.
