# 002 Clean Eval After Continuation

## Route

If subtask 001 passes, evaluate the continuation checkpoint with the same
Task048 strict clean matrix before treating it as a usable gait checkpoint.

## Log

- 2026-08-12 Subtask 001 passed with accepted continuation checkpoint
  `outputs/task049/bridge_smoke_per_step_mlp_replay/task049_bridge_smoke_per_step_mlp_replay_env512_step24_iter10_mb1_seed4900107/model_9.pt`
  and train summary
  `.agent/task/task049-true-txl-bridge-continuation/task049_bridge_smoke_per_step_mlp_replay_env512_step24_iter10_mb1_seed4900107.json`.
- 2026-08-12 Tried `task041_sequence_txl_clean_eval.py` first with the Task048
  clean-bins task id. The wrapper rejected that route at preflight with
  `task_not_task039_true_txl_clean_train`; evidence is retained at
  `outputs/task049/eval_bridge_smoke_per_step_mlp_replay/eval_model9_strict_clean/true_txl_vx0p4.json`.
  This was a route mismatch, not a simulator/eval failure.
- 2026-08-12 Ran the same strict Task048 clean matrix used for the bridge:
  `OUTPUT_DIR=/home/xyzl/yubo/locomotion_rl/outputs/task049/eval_bridge_smoke_per_step_mlp_replay/eval_model9_strict_clean_task048_matrix NUM_ENVS=64 STEPS=360 TRIAL_LENGTH_S=2.0 SEED=4800301 GPU_ID=0 PY=/home/xyzl/yubo/locomotion_rl/.venv/bin/python bash .agent/task/task048-local-4090-previous-gait-reproduction/task048_eval_clean_matrix.sh true-txl /home/xyzl/yubo/locomotion_rl/outputs/task049/bridge_smoke_per_step_mlp_replay/task049_bridge_smoke_per_step_mlp_replay_env512_step24_iter10_mb1_seed4900107/model_9.pt`.
- 2026-08-12 Clean matrix summary:
  `outputs/task049/eval_bridge_smoke_per_step_mlp_replay/eval_model9_strict_clean_task048_matrix/true-txl_clean_matrix_summary.json`
  with `pass=true`, `matrix_pass=true`, `case_passes=[true, true, true]`,
  and exact zero final-trial fall ratio for all three speeds.
- 2026-08-12 Final-trial metrics:
  command `0.4 m/s`: actual `0.332 m/s`, linear error `0.137`, yaw error
  `0.050`, gravity-xy max `0.030`, root-z min `0.781`, fall ratio `0.0`.
  Command `1.2 m/s`: actual `0.846 m/s`, linear error `0.379`, yaw error
  `0.060`, gravity-xy max `0.081`, root-z min `0.768`, fall ratio `0.0`.
  Command `2.0 m/s`: actual `1.171 m/s`, linear error `0.838`, yaw error
  `0.102`, gravity-xy max `0.111`, root-z min `0.753`, fall ratio `0.0`.
  The Task048 per-speed linear-error ceilings are `0.25/0.55/0.90`, so every
  case passed.

## Review

Status: passed. The continuation checkpoint satisfies the Task048 strict clean
matrix and does not regress the bridge zero-fall gate. This remains a short
continuation validation, not a claim of superiority over the Task048 bridge.
