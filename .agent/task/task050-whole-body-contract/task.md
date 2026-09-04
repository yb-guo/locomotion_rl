# Task 050: Whole-Body Contract

## Route

Freeze the 45-slot whole-body schema, task-side 193D observation contract,
double-reset semantics, named-robot mappings, and dependency-neutral rollout
types before connecting training code.

## Log

- 2026-08-19: Opened after the plan was corrected from a fixed G1/Go2 design
  to a procedural whole-body biped/quadruped distribution.
- 2026-08-19: Added `whole_body_v1_45`, `WholeBodyStep`,
  `WholeBodyPolicyOutput`, `WholeBodyRolloutBatch`, and task contract modules.

## Review

Verified: `tests/test_whole_body_contract.py` and
`tests/test_component_architecture.py` pass; the full repository suite passes
without Task048 changes.  The task spec embeds schema version/hash and the
pinned Unitree/Menagerie revisions.  Task048's saved checkpoint/eval artifacts
remain under `outputs/task048/normal_walk`.
