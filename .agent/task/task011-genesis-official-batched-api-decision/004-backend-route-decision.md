# Route

Task: task011-genesis-official-batched-api-decision

Goal: write the decision report that maps Franka, Go2, and G1 evidence to the
next backend route.

Scope:

- Summarize environment/version evidence.
- Summarize Franka official batched baseline.
- Summarize Go2 official locomotion baseline.
- Summarize G1 target asset/backend probe.
- Apply stop rules and classify blockers.
- Select the next task route.

Environment:

- local documentation update
- remote evidence must come from guarded H200 runs

Verify:

- report contains all required throughput metric definitions and observed
  values for attempted loops.
- report distinguishes build performance from steady-state performance.
- report distinguishes Genesis/H200 environment blockers from G1 asset/backend
  blockers.
- top-level task remains pending unless all required subtasks and review pass.

No Hack:

- Do not claim Genesis is unsuitable if only G1 failed after official assets
  passed.
- Do not claim G1 is suitable without tensor device and selected reset evidence.
- Do not hide skipped subtasks; cite the stop rule that skipped them.

Hardware: local docs plus H200 evidence from prior subtasks.

# Log

pending

# Review

Status: pending.
