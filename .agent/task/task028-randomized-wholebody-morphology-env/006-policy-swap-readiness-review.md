# 006: Policy Swap Readiness Review

## Route

Decide whether the environment contract is stable enough to start a
LocoFormer-style policy experiment. This is a review gate, not a policy
implementation task.

## Minimal Closed Loop

Feedback loop:

1. Read the final env contract, asset inspect JSON, MLP training logs,
   randomization curriculum results, and eval/render artifacts.
2. Produce a readiness report that answers whether policy work can start.
3. If ready, define the next policy task's minimum interfaces.

Pass:

- Fixed-topology MLP baseline passes deterministic and randomized eval.
- Obs/action contract is documented and stable.
- Randomization toggles are isolated and reproducible.
- Known failure modes are recorded.
- The next policy task can be expressed as a policy-only change.

Fail:

- MLP baseline has not passed.
- Env/reward/randomization is still changing in ways that alter obs/action
  shape.
- There is no eval evidence from saved checkpoints.
- The proposed LocoFormer policy would need to compensate for unresolved env
  bugs.

Evidence:

- Readiness report under
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task028/policy_swap_readiness/`
  plus updates to this subtask doc.

## Log

- 2026-05-19 Opened during diagnose audit to prevent premature policy
  complexity.

## Review

Status: planned.
