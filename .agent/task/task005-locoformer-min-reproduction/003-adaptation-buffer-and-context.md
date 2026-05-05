# Route

Task: task005-locoformer-min-reproduction

Goal: Add a bounded adaptation/context buffer.

Scope:

- `src/h200_locomotion_lab/agents`
- training integration
- tests for reset and episode boundary behavior

Verify:

- Context resets correctly per environment.
- Buffer length is bounded.
- Missing history is masked.

Environment:

- local tests

No Hack:

- no unbounded memory growth
- no cross-episode leakage
- no hidden batch dimension assumptions

Hardware:

- context buffer must be batch-friendly

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

