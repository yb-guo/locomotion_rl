# 003 Smoke And Overhead

## Route

Run:

- local agent inventory and migration tests;
- H200 env64 smoke for AdaptK4;
- H200 env8192 one-iteration overhead for AdaptK4;
- H200 registration check for GRU/token eval task ids.

Acceptance:

- construction and one PPO iteration complete;
- actor input dimensions are recorded;
- no CPU/Python history bottleneck is observed.

## Log

- 2026-05-28 Local validation passed with `PYTHONPATH=src`:
  `python -m h200_locomotion_lab.tools.inspect_agent`,
  `python -m pytest -p no:cacheprovider tests/test_agent_inventory.py tests/test_task033_history_buffer.py`
  (`4 passed, 5 skipped`), JSON parse for `task036_summary.json`, and AST
  parse for changed Python/scripts.
- 2026-05-28 H200 AdaptK4 env64 one-iteration smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task036/policy_train/036_adapt_smoke_env64_iter1_gpu0_seed3603602.stdout.log`.

## Review

Status: local and env64 smoke passed. Env8192 overhead is still pending.
