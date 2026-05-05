# Route

Task: task005-locoformer-min-reproduction

Goal: Freeze the observation/action schema before implementing transformer policy.

Scope:

- `configs/agents/locoformer_min.yaml`
- `docs/agent_submodules.md`
- environment adapter docs

Verify:

- Observation tensor fields, action fields, history fields, and units are listed.

Environment:

- local docs/code

No Hack:

- no changing schema silently during training
- no mixing world/body/joint frames without labels

Hardware:

- schema should support batched H200 training

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

