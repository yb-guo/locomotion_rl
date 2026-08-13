# 001 Bridge Continuation Smoke

## Route

Run a short `task041_sequence_txl_clean_train.py` PPO continuation from the
Task048 verified bridge checkpoint. This is a correctness and regression gate,
not a quality claim.

Command shape:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m \
  h200_locomotion_lab.tools.task041_sequence_txl_clean_train \
  --resume-checkpoint outputs/task048/bridge/model_task048_true_txl_bridge_from1558.pt \
  --num-envs 512 \
  --rollout-steps 24 \
  --iterations 10 \
  --save-interval 5 \
  --num-mini-batches 1 \
  --num-learning-epochs 2 \
  --learning-rate 0.000003 \
  --seed 4900101 \
  --device cuda:0 \
  --run-name task049_bridge_smoke_env512_step24_iter10_mb1_seed4900101 \
  --experiment-name task049_true_txl_bridge_continuation \
  --log-dir outputs/task049/bridge_smoke/task049_bridge_smoke_env512_step24_iter10_mb1_seed4900101 \
  --output-json .agent/task/task049-true-txl-bridge-continuation/task049_bridge_smoke_env512_step24_iter10_mb1_seed4900101.json
```

## Log

- 2026-08-12 Ran the original 10-iteration command with seed `4900101`.
  Evidence:
  `.agent/task/task049-true-txl-bridge-continuation/task049_bridge_smoke_env512_step24_iter10_mb1_seed4900101.json`.
  The run wrote checkpoint
  `outputs/task049/bridge_smoke/task049_bridge_smoke_env512_step24_iter10_mb1_seed4900101/model_9.pt`
  and used `env_cfg_mode=train`, `env_episode_length_s=20.0`, but failed the
  gate with `train_pipeline_pass=false`,
  `failure_reasons=[algorithm_debug_logprob_parity_failed, algorithm_debug_logprob_error_too_high, algorithm_debug_ratio_error_too_high]`,
  `max_logprob_abs_error=0.000171661376953125`, and
  `max_ratio_abs_error` above `1e-5`.
- 2026-08-12 Follow-up probes confirmed the old stored rollout logprob itself
  was exact: `max_stored_logprob_recompute_abs_error=0.0`. Distribution params
  showed the standard deviation matched, while actor mean replay differed by
  `~1e-6` to `3e-6`, enough to produce `~1e-4` summed logprob error across the
  31 action dimensions. The failed JSONs are retained under this task directory
  as seeds `4900102` through `4900105`.
- 2026-08-12 Fixed the replay shape mismatch by running the sequence replay MLP
  actor head once per rollout step with the same environment-batch shape used
  during collection. Targeted tests then passed:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m pytest -q -p no:cacheprovider tests/test_task040_sequence_txl_ppo_update_smoke.py tests/test_task041_sequence_txl_clean_train.py`
  -> `42 passed`.
- 2026-08-12 Passing 2-iteration proof:
  `.agent/task/task049-true-txl-bridge-continuation/task049_bridge_probe_per_step_mlp_replay_env512_step24_iter2_mb1_seed4900106.json`.
  It had `train_pipeline_pass=true`, checkpoint
  `outputs/task049/bridge_probe_per_step_mlp_replay/task049_bridge_probe_per_step_mlp_replay_env512_step24_iter2_mb1_seed4900106/model_1.pt`,
  non-empty rollout-start memory, per-step normalizer snapshot replay, and
  `max_logprob_abs_error=max_ratio_abs_error=0.0`.
- 2026-08-12 Passing 10-iteration bridge smoke:
  `.agent/task/task049-true-txl-bridge-continuation/task049_bridge_smoke_per_step_mlp_replay_env512_step24_iter10_mb1_seed4900107.json`.
  It had `train_pipeline_pass=true`, `failure_reasons=[]`,
  `resume_checkpoint=outputs/task048/bridge/model_task048_true_txl_bridge_from1558.pt`,
  `env_cfg_mode=train`, `env_episode_length_s=20.0`, checkpoint
  `outputs/task049/bridge_smoke_per_step_mlp_replay/task049_bridge_smoke_per_step_mlp_replay_env512_step24_iter10_mb1_seed4900107/model_9.pt`,
  `last_actor_normalizer_snapshot_count=24`, non-empty rollout-start memory,
  `max_distribution_param_abs_errors=[0.0, 0.0]`, and
  `max_logprob_abs_error=max_ratio_abs_error=0.0`.

## Review

Status: passed. The accepted continuation checkpoint for subtask 002 is
`outputs/task049/bridge_smoke_per_step_mlp_replay/task049_bridge_smoke_per_step_mlp_replay_env512_step24_iter10_mb1_seed4900107/model_9.pt`.
