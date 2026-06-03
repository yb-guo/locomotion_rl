# 005: Memory Causality Evidence

## Route

Evaluate trained Task044 candidates with the triplet contract. Positive evidence
requires normal quality pass plus material degradation in both zero-residual and
stateless-memory modes.

## Acceptance

- Normal, zero-residual, and stateless JSON paths are recorded.
- Triplet summary passes the Task044 contract.
- Source checkpoint, seed, speed, and hidden-fault schedule match.
- Review explicitly states what is proven and what is not proven.

## Log

- 2026-05-31 Planned after 004.
- 2026-05-31 Evaluated the 25-iteration hidden-fault checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_train/logs_model5349_hidden_fault_env1024_iter25_seed4400301/model_24.pt`.
- Normal:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/model5349_hidden_fault_env1024_iter25_model24_none_vx1p6_seed4400401.json`.
  `quality_gate_pass=false`; final trial completion was `1.0`, fall ratio was
  `0.0`, but `lin_vel_error.mean=0.4572296142578125` and regression checks
  failed.
- Zero residual:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/model5349_hidden_fault_env1024_iter25_model24_zero_residual_vx1p6_seed4400401.json`.
  Final `lin_vel_error.mean=0.4618203639984131`.
- Stateless memory:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/hidden_fault_eval/model5349_hidden_fault_env1024_iter25_model24_stateless_vx1p6_seed4400401.json`.
  Final `lin_vel_error.mean=0.4576954245567322`.
- Triplet summary:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task044/triplet_summary/model5349_hidden_fault_env1024_iter25_model24_triplet_seed4400401.json`.
  `task044_memory_required_pass=false` with
  `normal_quality_gate_not_passed`, `zero_residual_ablation_not_degraded`, and
  `stateless_memory_ablation_not_degraded`.

## Review

Status: failed, not accepted.

The residual path is active, but current hidden-fault schedules still do not
produce a memory-causality gap. The likely target-design issue is that final
trial evaluation can restart the failure timeline from t=0, so the final
2-second window may not require a remembered fault identity. Subtask 006 should
probe immediate hidden-fault onset before any more long training.
