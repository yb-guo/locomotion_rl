# Route

Task: task005-locoformer-min-reproduction

Goal: Implement the minimal transformer policy core.

Scope:

- `src/h200_locomotion_lab/agents/locoformer.py`
- tests for shape and causal/context behavior

Verify:

- Forward pass works on fake batched observations.
- Output action distribution and value head shapes match baseline.

Environment:

- local tests first
- H200 target later

No Hack:

- no architecture claim without test
- no hard-coded robot DOF unless explicitly scoped
- no hidden CPU tensor in training hot path

Hardware:

- avoid unnecessary host/device sync
- record memory for real runs

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

