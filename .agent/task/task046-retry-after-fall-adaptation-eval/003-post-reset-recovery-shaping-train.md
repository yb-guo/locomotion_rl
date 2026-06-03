# 003: Post-Reset Recovery Shaping Train

## Route

Improve speed of recovery after a physical inner reset without changing the
policy architecture.

Implementation contract:

- Add a default-off wrapper around the existing Task044 multi-trial runner.
- Actor inputs stay unchanged and fault identity remains hidden.
- TXL memory behavior stays unchanged: inner resets are recorded, outer resets
  clear memory.
- Shaping only applies on final trial after reset:
  - early recovery window: penalize forward velocity shortfall, projected
    gravity XY, and low root height;
  - tail window: penalize remaining forward velocity shortfall.
- Use existing retry eval to compare against the current `model_39.pt`.

Training route:

- First run H200 smoke: small env count, 1 iteration, wrapper enabled, prove
  train pipeline and wrapper debug samples.
- Then warm-start from current `model_39.pt` and run a short targeted stage.
- Evaluate the new checkpoint on the same 18-case retry matrix from subtask
  002.

## Acceptance

- Local pytest covers CLI cfg mutation and wrapper reward behavior.
- H200 train smoke writes JSON with `train_pipeline_pass=true` and nonzero
  `task046_post_reset_recovery_reward_debug.sample_count`.
- A trained checkpoint exists.
- The new checkpoint is evaluated against the subtask 002 matrix.
- Review compares:
  - final-trial fall counts;
  - final-trial full-window velocity error;
  - final-trial tail-window velocity error;
  - whether recovery is faster or merely equally stable.

## Log

- 2026-06-02 Opened after subtask 002 showed retry eval is stable enough to
  justify training-contract work.
- 2026-06-02 Local implementation:
  `Task046PostResetRecoveryRewardVecEnvWrapper` plus default-off CLI/train-cfg
  switches in the sequence-aware train path.
- 2026-06-02 Local validation:
  `python -m pytest -q -p no:cacheprovider tests\test_task041_sequence_txl_clean_train.py tests\test_task044_hidden_fault_target.py --tb=short --basetemp '.test_tmp_task046_recovery_local'`
  passed `18 passed, 8 skipped`.
- 2026-06-02 Local validation:
  `python -m h200_locomotion_lab.tools.inspect_agent` passed.
- 2026-06-02 H200 validation:
  `python -m pytest -q -p no:cacheprovider tests/test_task041_sequence_txl_clean_train.py tests/test_task044_hidden_fault_target.py --tb=short --basetemp /tmp/task046_recovery_pytest2`
  passed `26 passed`.
- 2026-06-02 H200 long smoke:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/post_reset_recovery_train/smoke_long/train_summary.json`.
  Result: `train_pipeline_pass=true`,
  `task046_post_reset_recovery_reward_debug.sample_count=561`,
  `recovery_sample_count=375`, `tail_sample_count=186`.
- 2026-06-02 Stage1 targeted train:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/post_reset_recovery_train/stage1_env1024_rollout220_iter20_lr5e6_seed4620402/train_summary.json`.
  Warm-started from current `model_39.pt`, 1024 envs, rollout `220`,
  `20` iterations, lr `5e-6`. Checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/post_reset_recovery_train/stage1_env1024_rollout220_iter20_lr5e6_seed4620402/logs/model_19.pt`.
  Result: `train_pipeline_pass=true`, final iteration `19`,
  wrapper `sample_count=1470376`, `reward_delta_mean=-0.3305`.
- 2026-06-02 Stage1 baseline-seed retry matrix:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_matrix/stage1_post_reset_recovery_model19_knee_onset_multiseed_v4_baseline_seeds/summary_corrected.json`.
  Comparison:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_matrix/stage1_post_reset_recovery_model19_knee_onset_multiseed_v4_baseline_seeds/comparison_to_model39.json`.
  Result: `18/18` pass, final falls mean `3.28`,
  full final velocity error mean `0.5845`, early-window velocity error mean
  `0.9112`, tail-window velocity error mean `0.2555`. Compared with
  `model_39.pt`, all `18/18` cases improved early and tail velocity error;
  mean deltas were final falls `-0.28`, full velocity error `-0.0261`,
  early velocity error `-0.0235`, tail velocity error `-0.0285`.
- 2026-06-02 Stage2 early-focused train:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/post_reset_recovery_train/stage2_early_env1024_rollout220_iter20_lr5e6_seed4620502/train_summary.json`.
  Warm-started from Stage1 `model_19.pt`, kept the same train scale, increased
  early velocity weight to `1.2`, reduced tail velocity weight to `0.15`, and
  set orientation weight to `0.25`. Checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/post_reset_recovery_train/stage2_early_env1024_rollout220_iter20_lr5e6_seed4620502/logs/model_19.pt`.
  Result: `train_pipeline_pass=true`, final iteration `19`,
  wrapper `sample_count=1470988`, `reward_delta_mean=-0.5992`.
- 2026-06-02 Stage2 baseline-seed retry matrix:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_matrix/stage2_early_model19_knee_onset_multiseed_baseline_seeds_v3/summary_corrected.json`.
  Comparisons:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_matrix/stage2_early_model19_knee_onset_multiseed_baseline_seeds_v3/comparison_to_model39.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_matrix/stage2_early_model19_knee_onset_multiseed_baseline_seeds_v3/comparison_to_stage1.json`.
  Result: `18/18` pass, final falls mean `2.83`,
  full final velocity error mean `0.5742`, early-window velocity error mean
  `0.9033`, tail-window velocity error mean `0.2430`. Compared with
  `model_39.pt`, all `18/18` cases improved early and tail velocity error;
  mean deltas were final falls `-0.72`, full velocity error `-0.0364`,
  early velocity error `-0.0315`, tail velocity error `-0.0410`. Compared
  with Stage1, all `18/18` cases still improved early and tail velocity error;
  mean deltas were final falls `-0.44`, full velocity error `-0.0103`,
  early velocity error `-0.0080`, tail velocity error `-0.0125`.

## Review

Status: evidence complete for subtask 003.

The default-off reward wrapper works and does not change actor inputs or expose
fault identity. Both targeted stages improved retry recovery under the same
18-case baseline-seed matrix. Stage2 is the best current checkpoint for the
retry-after-fall contract:

- `model_39.pt`: `18/18` pass, final falls mean about `3.56`, full velocity
  error about `0.6105`, tail velocity error about `0.2840`.
- Stage1 `model_19.pt`: `18/18` pass, final falls mean `3.28`, full velocity
  error `0.5845`, early velocity error `0.9112`, tail velocity error `0.2555`.
- Stage2 `model_19.pt`: `18/18` pass, final falls mean `2.83`, full velocity
  error `0.5742`, early velocity error `0.9033`, tail velocity error `0.2430`.

Do not overclaim this as continuous deployment stability. The old no-physical-
reset gate remains separate and still unresolved. Also, even Stage2 only
improves the first post-reset second modestly; the larger gain is in the final
trial tail window. Further progress likely needs a policy/training-contract
handoff rather than more local reward tuning.
