# Route

Task: task006-sonic-genesis-action-policy

Goal: Connect the real SONIC policy forward path to the Genesis 29-motor G1
backend after action replay passes.

Dependency:

- `001-genesis-action-replay.md` must pass first.

Pass condition:

- SONIC policy input/output contract is documented.
- One policy forward pass runs and returns a 29D action compatible with the
  Genesis action contract.
- A short H200 Genesis rollout runs from policy actions without non-finite state.
- Base height, action range, qvel, and termination/failure state are recorded.
- Produce a short GIF/contact sheet if the rollout smoke passes.

Fail condition:

- Missing or unusable policy artifact.
- Unclear observation/history/command contract.
- Policy output is not 29D or cannot be mapped to the G1 29-motor contract.
- Genesis rollout becomes non-finite or falls outside smoke range.

Implementation plan:

1. Locate SONIC deploy policy entrypoint and policy I/O definitions.
2. Run policy forward outside Genesis with a controlled dummy/recorded input.
3. Map the 29D policy output into `GenesisG1Env.step(action)`.
4. Run short H200 rollout.
5. Render GIF/contact sheet only after numeric smoke passes.

# Log

- 2026-05-07: Opened route as L2. Blocked until action replay L1 passes.
- 2026-05-07: L1 action replay passed. This route is now unblocked, but no
  SONIC policy inspection or rollout has been run yet.

# Review

Status: open.
