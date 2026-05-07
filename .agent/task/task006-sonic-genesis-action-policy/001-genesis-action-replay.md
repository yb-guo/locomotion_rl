# Route

Task: task006-sonic-genesis-action-policy

Goal: Prove that the validated Genesis 29-motor G1 backend can be driven by
29D normalized action sequences through `GenesisG1Env.step(action)`.

Pass condition:

- H200 run completes without non-finite state.
- Action shape is exactly 29.
- Actions are clipped or validated in the same contract used by
  `GenesisG1Env.step`.
- Base height stays inside the configured smoke range.
- Max qvel and action range are recorded.
- Produce a short GIF/contact sheet if the smoke passes.

Fail condition:

- Any non-finite state.
- Action dimension mismatch.
- Robot falls below the smoke height range.
- Genesis scene build or step fails.
- H200 SSH/session failure before the command starts; record as infra failure,
  not simulator failure.

Implementation plan:

1. Add a small action replay tool that accepts either:
   - a CSV containing one 29D action row per policy step; or
   - a deterministic built-in action fixture for smoke testing.
2. Run local tests for parser/contract behavior without importing Genesis.
3. Sync to H200.
4. Run a short H200 smoke through `GenesisG1Env.step(action)`.
5. If the numeric smoke passes, render a short dynamic GIF/contact sheet.

# Log

- 2026-05-07: Opened route. This subtask is the next step requested by the
  user. It intentionally precedes SONIC policy integration.

# Review

Status: open.

Do not mark passed without H200 log evidence and local test evidence.
