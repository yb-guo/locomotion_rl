# 001: First-Principles Boundary

## Route

Start from what can change independently: physics implementation, MDP
semantics, action-distribution parameterization, and learning rule. Assign each
piece of state to exactly one owner and reject task/algorithm cross-imports.

## Log

- 2026-08-19 Defined task as observation/action/transition/reward/reset/
  termination/metrics, with device and optimizer concerns explicitly excluded.
- 2026-08-19 Defined MIP/flow/diffusion/JiT as policy families; their RL update
  is selected separately through declared capabilities.
- 2026-08-19 Recorded the decision and legacy migration map in
  `.agent/doc/component_architecture.md`.

## Review

Status: passed. Every concept has one owner and the dependency graph has a
mechanical test target.
