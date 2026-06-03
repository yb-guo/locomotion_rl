# 005: LocoFormer Reproduction Handoff

## Route

Summarize what Task045 proved even if it does not close the continuous
left-knee stability gate. The handoff should distinguish reusable eval/task
contract pieces from the policy architecture work needed for the next
LocoFormer reproduction task.

## Acceptance

- Identify whether a passing checkpoint exists.
- Identify reusable task/eval contract pieces.
- Identify remaining LocoFormer gaps without claiming reproduction.

## Log

- 2026-06-02 Negative handoff: no checkpoint passed the unchanged old
  continuous gate. The best checkpoint remains
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task045/hidden_fault_train/logs_long_survival_all_env2048_iter40_lr5e6_seed4520402/model_39.pt`,
  but repeated old-gate diagnostic evals still show physical reset events and
  post-fault fall ratios above `0.05`.
- 2026-06-02 Reusable pieces for the next LocoFormer task:
  - unchanged continuous old-gate eval command and JSON contract;
  - `physical_reset_time_diagnostic` for phase/bin failure attribution;
  - evidence that immediate-left-knee training-task pass is not enough for a
    2.0 s hidden-onset old gate;
  - evidence that both full-actor and memory-branch-only local PPO repair
    degrade or fail to improve the old gate.
- 2026-06-02 Remaining policy gap: the current true-TXL/adaptation branch does
  not reliably convert hidden fault onset history into stable post-fault gait.
  The next task should change policy mechanism/training contract, not continue
  left-knee-specific reward tuning.

## Review

Status: negative handoff complete.

Task045 does not provide a passing LocoFormer baseline. It provides a bounded
negative result and a sharper failure diagnostic for the next policy task.
