# 003: H200 Convergence Train Eval Loop

## Route

Run H200 train/eval loops from the current best checkpoint until normal
continuous left-knee eval passes or the route produces a documented negative
result. Normal quality must pass before memory ablation triplets are promoted.

## Acceptance

- H200 training JSONs record pipeline pass/failure.
- Continuous normal eval JSONs record pass/failure and post-fault metrics.
- The task does not mark passed from train success alone.

## Log

- 2026-06-02 Survival continuation ran on H200 but did not pass continuous
  eval. Best short-survival model39 remained above the gate with physical reset
  events still present.
- 2026-06-02 Long-survival continuation ran on H200. Current best is
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/hidden_fault_train/logs_long_survival_all_env2048_iter40_lr5e6_seed4520402/model_39.pt`.
  Its continuous eval JSON is
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/continuous_fault_eval/long_survival_env2048_lr5e6_model39_continuous_normal_left_knee_seed4520501.json`.
  It improved the post-fault fall ratio to `0.078125`, but still failed with
  `pipeline_pass=false`, `quality_gate_pass=false`, and physical reset events.
- 2026-06-02 Reset-time diagnostic plus two targeted continuation stages ran on
  H200. Neither full-actor old-gate-aligned long-tail training nor
  scope-limited memory-branch training improved the unchanged old gate. Best
  evidence remains the original long-survival checkpoint rather than the
  targeted continuations.

## Review

Status: negative for local convergence.

No Task045 H200 convergence run passed the unchanged old continuous gate.
Because the agreed diagnostic plus one or two targeted stages have been used,
the local reward/curriculum repair route should stop here and hand off to the
next LocoFormer policy task.
