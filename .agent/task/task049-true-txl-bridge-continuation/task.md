# Task 049: True-TXL Bridge Continuation

## Route

Continue True-TXL PPO from the verified Task048 AdaptK160 bridge instead of
scratch training or the collapsed Task047 `model_499.pt` lineage.

Source checkpoint:

- `outputs/task048/bridge/model_task048_true_txl_bridge_from1558.pt`

Execution gates:

- Use local RTX 4090 + MJLab/MuJoCo Warp only.
- Use `task041_sequence_txl_clean_train.py` so the train config is loaded with
  `play=False` and the JSON records `env_episode_length_s`.
- Start with a short parity smoke from the bridge checkpoint.
- Promote a continuation checkpoint only when `train_pipeline_pass=true`,
  strict sequence replay parity passes, and a saved clean fixed-speed eval does
  not regress the Task048 bridge zero-fall result.
- Do not use Task047 `model_499.pt`.
- Do not download checkpoints, robot assets, datasets, or upstream repos.

## Planned Slices

1. `001-bridge-continuation-smoke.md`
   - Run a short PPO continuation from the Task048 bridge checkpoint.
   - Require `env_cfg_mode=train`, `episode_length_s=20`, and strict logprob
     parity.

2. `002-clean-eval-after-continuation.md`
   - Evaluate the new checkpoint at fixed clean commands.
   - Compare against the Task048 bridge matrix before making any quality claim.

## Acceptance Criteria

- The short continuation writes a checkpoint and JSON evidence.
- Training JSON has `resume_checkpoint` equal to the Task048 bridge checkpoint,
  `env_cfg_mode=train`, and non-play `env_episode_length_s`.
- Strict Task047 logprob/ratio replay parity passes under non-empty rollout
  memory.
- Clean eval evidence exists for the continuation checkpoint before it is
  called usable for gait.
- Any failed parity or eval is recorded as a blocker, not promoted.

## Log

- 2026-08-12 Opened at user request to continue True-TXL after Task048 passed
  at the verified bridge. Current local GPU check showed NVIDIA GeForce RTX
  4090 with about 45.9 GB free. No external checkpoint, dataset, asset, or
  upstream repo download was requested or performed.
- 2026-08-12 Initial continuation attempts from the Task048 bridge wrote
  checkpoints but failed strict sequence replay parity. The first 10-iteration
  smoke
  `.agent/task/task049-true-txl-bridge-continuation/task049_bridge_smoke_env512_step24_iter10_mb1_seed4900101.json`
  had `train_pipeline_pass=false`,
  `max_logprob_abs_error=0.000171661376953125`, and
  `max_ratio_abs_error` above the `1e-5` gate. Follow-up probes showed the
  stored rollout logprob recomputed exactly from stored distribution params,
  while sequence replay actor means differed at about `1e-6` to `3e-6`.
- 2026-08-12 Added deterministic replay alignment in
  `src/h200_locomotion_lab/training/rsl_history_wrapper.py`: disabled
  mismatched attention/TF32 fastpaths for Task040 replay diagnostics, recorded
  detailed distribution-param parity diagnostics, disabled rollout
  `inference_mode`, and then changed the sequence replay actor head to run the
  MLP per rollout step with the same `[env_batch, latent]` shape used during
  collection. Targeted tests:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m pytest -q -p no:cacheprovider tests/test_task040_sequence_txl_ppo_update_smoke.py tests/test_task041_sequence_txl_clean_train.py`
  passed with `42 passed`.
- 2026-08-12 Passing 2-iteration proof after the per-step MLP replay change:
  `.agent/task/task049-true-txl-bridge-continuation/task049_bridge_probe_per_step_mlp_replay_env512_step24_iter2_mb1_seed4900106.json`
  with `train_pipeline_pass=true`, checkpoint
  `outputs/task049/bridge_probe_per_step_mlp_replay/task049_bridge_probe_per_step_mlp_replay_env512_step24_iter2_mb1_seed4900106/model_1.pt`,
  `env_cfg_mode=train`, `env_episode_length_s=20.0`,
  non-empty rollout-start memory, per-step normalizer replay snapshots, and
  `max_logprob_abs_error=max_ratio_abs_error=0.0`.
- 2026-08-12 Passing 10-iteration bridge continuation gate:
  `.agent/task/task049-true-txl-bridge-continuation/task049_bridge_smoke_per_step_mlp_replay_env512_step24_iter10_mb1_seed4900107.json`
  with `train_pipeline_pass=true`, `failure_reasons=[]`, checkpoint
  `outputs/task049/bridge_smoke_per_step_mlp_replay/task049_bridge_smoke_per_step_mlp_replay_env512_step24_iter10_mb1_seed4900107/model_9.pt`,
  `resume_checkpoint=outputs/task048/bridge/model_task048_true_txl_bridge_from1558.pt`,
  `env_cfg_mode=train`, `env_episode_length_s=20.0`,
  `last_actor_normalizer_snapshot_count=24`, non-empty rollout-start memory,
  `max_distribution_param_abs_errors=[0.0, 0.0]`, and
  `max_logprob_abs_error=max_ratio_abs_error=0.0`.
- 2026-08-12 Passing Task048 strict clean matrix on the continuation
  checkpoint:
  `outputs/task049/eval_bridge_smoke_per_step_mlp_replay/eval_model9_strict_clean_task048_matrix/true-txl_clean_matrix_summary.json`
  with `matrix_pass=true`, `case_passes=[true, true, true]`, and zero final
  falls at 0.4, 1.2, and 2.0 m/s. Final-trial actual forward speeds were
  `0.332/0.846/1.171 m/s`; final-trial linear velocity errors were
  `0.137/0.379/0.838 m/s`, all within the Task048 thresholds
  `0.25/0.55/0.90 m/s`. This is a usable short continuation checkpoint, not a
  superiority claim over the Task048 bridge.

## Review

Status: passed. Task049 has a verified short True-TXL PPO continuation from the
Task048 bridge with strict replay parity and a saved zero-fall Task048 clean
matrix. No Task047 collapsed checkpoint, external checkpoint, dataset, asset, or
upstream repo was used.
