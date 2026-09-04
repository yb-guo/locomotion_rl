# 003: Dependency Guard and Handoff

## Route

Prove the boundary with AST dependency checks, config inspection, a fake
vectorized task/policy/algorithm interaction, and the full repository
regression suite. Keep historical modules working and record their migration
destinations.

## Log

- 2026-08-19 Added `tests/test_component_architecture.py` and the simulator-free
  `python -m h200_locomotion_lab.tools.inspect_components` inspection command.
- 2026-08-19 Focused task/policy/algorithm suite passed (`24 passed`), including
  identity checks proving legacy imports delegate rather than duplicate.
- 2026-08-19 Full repository regression passed (`724 passed`, 35 upstream
  TorchScript deprecation warnings). Focused full-rule Ruff, repository
  critical Ruff, `inspect_agent`, `inspect_components`, and
  `git diff --check` all passed.

## Review

Status: passed. Dependency direction, config ownership, composition failure,
generic interaction, concrete migration, and compatibility are all covered by
executable evidence.
