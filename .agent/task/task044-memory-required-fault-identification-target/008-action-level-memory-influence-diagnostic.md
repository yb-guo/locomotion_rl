# 008: Action-Level Memory Influence Diagnostic

## Route

Subtasks 005, 006, and 007 all failed with tied rollout metrics across normal,
zero-residual, and stateless-memory modes. The next smallest diagnostic is to
inspect policy output, not reward:

- if action statistics are tied, the TXL memory path is not materially changing
  control;
- if action statistics differ but locomotion metrics remain tied, memory changes
  actions but the training target did not make those changes useful.

This subtask only diagnoses the bottleneck. It cannot mark Task044 passed.

Implementation:

- record action statistics in `task037_multitrial_eval_checkpoint.py`;
- attach per-trial `action_stats` and windowed `action_stats` when
  `--final-window-s` is used;
- add a Task044 action-influence summary CLI for normal / zero-residual /
  stateless eval JSONs;
- keep no-overclaim fields explicit.

## Acceptance

- Local tests prove action stats are accumulated with masks.
- Local tests prove the action-influence summary reports tied actions as a
  diagnostic failure.
- Local tests prove nontrivial action deltas are recorded without claiming
  memory causality or LocoFormer reproduction.
- H200 probe records normal, zero-residual, stateless eval JSONs with
  `action_stats` and one action-influence summary JSON.

## Log

- 2026-05-31 Created after the 0.5s windowed final-trial probe still showed
  tied behavior across memory ablations.
- 2026-05-31 Implemented masked action statistics in
  `src/h200_locomotion_lab/tools/task037_multitrial_eval_checkpoint.py`.
  Eval JSONs now attach `action_stats` to every trial and to
  `final_trial_window` when `--final-window-s` is used.
- 2026-05-31 Added pure action-influence contract and CLI:
  `src/h200_locomotion_lab/training/task044_action_influence_contract.py`
  and
  `src/h200_locomotion_lab/tools/task044_action_influence_summary.py`.
  The summary is diagnostic-only and keeps `memory_causality_claim=false`.
- 2026-05-31 Local validation passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task037_mjlab_smoke_scripts.py tests\test_task044_action_influence_contract.py tests\test_task044_action_influence_summary_cli.py --tb=short`
  with 15 passed and 1 skipped. The skipped test is the local torch-dependent
  accumulator test on a machine without torch.
- 2026-05-31 CLI help passed:
  `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task044_action_influence_summary --help`.
- 2026-05-31 H200 action-stat probe completed after fixing the remote
  `PYTHONPATH` to include both the adapter repo and the external MJLab repo.
  Summary JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/action_influence_summary/model5349_hidden_fault_env1024_iter25_model24_actionstats_left_knee_joint_vx1p6_seed4400801.json`.
  Source eval JSONs:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/model5349_hidden_fault_env1024_iter25_model24_actionstats_none_left_knee_joint_vx1p6_seed4400801.json`,
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/model5349_hidden_fault_env1024_iter25_model24_actionstats_zero_left_knee_joint_vx1p6_seed4400801.json`,
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/model5349_hidden_fault_env1024_iter25_model24_actionstats_stateless_left_knee_joint_vx1p6_seed4400801.json`.
- 2026-05-31 H200 result:
  `action_influence_detected=true` only because zero-residual changed action
  stats slightly. `stateless_memory_action_stats_tied` remained in
  `failure_reasons`. Windowed action deltas were:
  zero-residual `mean_abs_l1_delta=0.009924508390888091` and
  `mean_l2_delta=-0.023755788803100586`; stateless-memory
  `mean_abs_l1_delta=0.00084640410157942` and
  `mean_l2_delta=0.0009143352508544922`.
  Windowed rollout metrics also remained tied: normal
  `lin_vel_error.mean=1.213126540184021`, zero-residual
  `1.211687684059143`, and stateless-memory `1.2127338647842407`.

## Review

Status: failed diagnostically, closed as evidence.

The current `model_24` Task044 hidden-fault checkpoint does not use stateful
TXL memory in a behaviorally meaningful way. Zeroing the residual changes the
action distribution slightly, but replacing stateful memory with stateless
memory leaves actions effectively tied. This points away from "eval metric hid
the effect" and toward "the policy/training setup is not using cross-trial
memory." The next route should change the training target or policy coupling so
stateful memory has a supervised/advantaged reason to affect action selection.
