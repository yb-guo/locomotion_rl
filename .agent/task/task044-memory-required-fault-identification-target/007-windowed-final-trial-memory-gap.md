# 007: Windowed Final-Trial Memory Gap

## Route

Subtask 006 showed that whole-final-trial metrics still keep normal,
zero-residual, and stateless-memory behavior tied. The next redesign is a
diagnostic, not a success claim: collect metrics over only the first N seconds
of the final trial so slow within-trial re-identification cannot hide whether
memory helps at failure onset.

Implementation:

- add `--final-window-s` to the Task037 multitrial eval CLI;
- record `final_trial_window` beside the existing `final_trial`;
- keep default behavior unchanged when `--final-window-s` is not provided;
- add a Task044 triplet summary `--metric-scope final_trial_window` option for
  degradation deltas only;
- keep the normal quality gate unchanged, so windowed deltas cannot create a
  false pass if locomotion quality fails.

## Acceptance

- Local tests prove default triplet behavior still reads `final_trial`.
- Local tests prove `metric_scope=final_trial_window` changes ablation deltas.
- CLI help shows `--final-window-s` and `--metric-scope`.
- H200 probe records normal, zero-residual, stateless, and triplet summary JSONs
  for immediate hidden left-knee failure with `--final-window-s 0.5`.
- Review states whether the windowed target creates a real memory gap.

## Log

- 2026-05-31 Created after subtask 006 failed with tied full-trial ablations.
- 2026-05-31 Implemented local support:
  - `src/h200_locomotion_lab/tools/task037_multitrial_eval_checkpoint.py`
    now accepts `--final-window-s` and records `final_trial_window`;
  - `src/h200_locomotion_lab/tools/task041_sequence_txl_clean_eval.py`
    forwards `--final-window-s` through Task042/Task044 wrappers;
  - `src/h200_locomotion_lab/training/task044_memory_required_contract.py`
    accepts `Task044TripletThresholds(metric_scope="final_trial_window")`;
  - `src/h200_locomotion_lab/tools/task044_memory_required_triplet_summary.py`
    accepts `--metric-scope final_trial_window`.
- 2026-05-31 Local validation passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task037_mjlab_smoke_scripts.py tests\test_task041_sequence_txl_clean_eval.py tests\test_task044_memory_required_contract.py tests\test_task044_triplet_summary_cli.py tests\test_task044_hidden_fault_target.py --tb=short --basetemp C:\Users\guoyubo.9\AppData\Local\Temp\pytest_task044_windowed_2`
  with 31 passed.
- 2026-05-31 CLI help showed `--final-window-s` on
  `h200_locomotion_lab.tools.task037_multitrial_eval_checkpoint` and
  `h200_locomotion_lab.tools.task044_hidden_fault_eval`; triplet CLI help
  showed `--metric-scope {final_trial,final_trial_window}`.
- 2026-05-31 H200 windowed normal eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/model5349_hidden_fault_env1024_iter25_model24_window0p5_left_knee_none_vx1p6_seed4400701.json`.
  `final_trial_window.lin_vel_error.mean=1.2125095129013062`;
  full final trial `lin_vel_error.mean=0.4610653817653656`;
  `quality_gate_pass=false`.
- 2026-05-31 H200 windowed zero-residual eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/model5349_hidden_fault_env1024_iter25_model24_window0p5_left_knee_zero_vx1p6_seed4400701.json`.
  `final_trial_window.lin_vel_error.mean=1.2107001543045044`.
- 2026-05-31 H200 windowed stateless-memory eval:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/model5349_hidden_fault_env1024_iter25_model24_window0p5_left_knee_stateless_vx1p6_seed4400701.json`.
  `final_trial_window.lin_vel_error.mean=1.2120933532714844`.
- 2026-05-31 H200 windowed triplet summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/model5349_hidden_fault_env1024_iter25_model24_window0p5_left_knee_triplet_seed4400701.json`.
  Result: `task044_memory_required_pass=false` with
  `normal_quality_gate_not_passed`, `zero_residual_ablation_not_degraded`, and
  `stateless_memory_ablation_not_degraded`. Window deltas were
  `-0.0018093585968017578` for zero-residual and
  `-0.00041615962982177734` for stateless-memory.

## Review

Status: failed, not accepted.

The windowed metric is implemented and useful diagnostically, but the current
checkpoint/target still does not show memory causality. The first 0.5 seconds
of the final trial are effectively tied across normal, zero-residual, and
stateless-memory modes. The next route should inspect or change the training
signal so TXL memory produces action-level differences when hidden motor state
is useful.
