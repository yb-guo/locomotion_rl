# 003 Clean Matrix And Bridge

## Route

Evaluate each candidate at fixed forward commands 0.4, 1.2, and 2.0 m/s using
the deterministic three-trial runner. Require zero final-trial falls in every
case before migration. Once AdaptK160 passes, construct the Task041 bridge and
evaluate it before any PPO continuation.

## Log

- Added `task048_eval_clean_matrix.sh` for `mlp`, `adaptk4`, `adaptk160`, and
  `true-txl` checkpoint shapes.
- The matrix requires final fall ratio exactly zero and uses per-speed linear
  tracking ceilings `0.25/0.55/0.90 m/s` for commands `0.4/1.2/2.0 m/s`.
  These admit the historical `model_5467.pt` metrics while rejecting stable
  standing at low speed. Each raw eval JSON is retained.
- Accepted MLP `model_1200.pt` matrix:
  `outputs/task048/mlp_speed_bins/eval_model1200/mlp_clean_matrix_summary.json`;
  actual forward speeds `0.318/0.800/1.132 m/s`, errors
  `0.149/0.412/0.876 m/s`, zero falls.
- Accepted AdaptK4 `model_1319.pt` matrix:
  `outputs/task048/adaptk4/eval_model1319/adaptk4_clean_matrix_summary.json`;
  actual forward speeds `0.300/0.833/1.167 m/s`, errors
  `0.155/0.383/0.839 m/s`, zero falls.
- Accepted AdaptK160 `model_1558.pt` matrix:
  `outputs/task048/adaptk160/eval_model1558/adaptk160_clean_matrix_summary.json`;
  actual forward speeds `0.340/0.850/1.162 m/s`, errors
  `0.135/0.374/0.847 m/s`, zero falls.
- Bridge report:
  `outputs/task048/bridge/model_task048_true_txl_bridge_from1558.json` with
  `warmstart_pipeline_pass=true`, 17 copied actor keys, 12 copied critic keys,
  and 17 fresh TXL-specific keys.
- Accepted strict-clean True-TXL bridge matrix:
  `outputs/task048/bridge/eval_true_txl_clean_from1558/true-txl_clean_matrix_summary.json`;
  actual forward speeds `0.340/0.850/1.162 m/s`, errors
  `0.135/0.374/0.847 m/s`, zero falls. Its task is the dedicated Task048 clean
  registration, not the randomized Task038 train registration.

## Review

Status: passed. Every migration candidate and the final bridge has persisted
three-speed JSON evidence satisfying the exact zero-fall and tracking gates.
No True-TXL PPO update was run or claimed because the independent Task047
strict replay-parity gate remains open.
