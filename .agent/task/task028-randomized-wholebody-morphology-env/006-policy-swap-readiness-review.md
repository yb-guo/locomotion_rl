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
- 2026-05-19 Reviewed completed 001-005 evidence. The fixed-topology gripper
  environment has stable obs/action shape, staged randomization toggles,
  MLP PPO smoke, deterministic eval pass, randomized holdout eval pass, and
  render evidence.
- 2026-05-19 Wrote readiness report:
  `.agent/task/task028-randomized-wholebody-morphology-env/artifacts/task028_policy_swap_readiness_report.md`.

## Review

Status: passed.

The environment is ready for a next policy-only fixed-topology experiment. The
recommended next task is to keep the `Unitree-G1-Gripper-Flat-*` env family and
the 31-action output contract unchanged, then replace only the policy model or
observation encoder. A LocoFormer-style first pass should decode into the same
flat action vector and should compare directly against the MLP
`model_600.pt` baseline.

Not ready in this task: variable topology, variable DoF, padding/masks, full
dexterous hands, generic morphology asset generation, gripper manipulation
contacts, delay/smoothing randomization, and multi-term ONNX export metadata.
