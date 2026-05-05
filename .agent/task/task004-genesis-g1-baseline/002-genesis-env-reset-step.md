# Route

Task: task004-genesis-g1-baseline

Goal: Implement or wrap a Genesis env that can reset and step.

Scope:

- `src/h200_locomotion_lab/envs/genesis_adapter.py`
- tests for reset/step boundary where possible

Verify:

- Minimal script resets and steps without training.

Environment:

- Linux H200 target for real Genesis
- local stub tests allowed

No Hack:

- no simulator import at module import time if it breaks local tests
- no global mutable singleton scene
- no unbounded per-step Python logging

Hardware:

- avoid CPU/GPU sync in hot path
- batch envs when real training starts

# Log

# Review

Result: pending
Syntax:
Hack:
Scope:
Efficiency:
Hardware:
Verify:
Findings:

