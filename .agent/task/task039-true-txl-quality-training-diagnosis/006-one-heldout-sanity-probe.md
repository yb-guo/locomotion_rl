# 006 One Heldout Sanity Probe

## Route

After clean train-variant quality improves, run one held-out G1-like morphology
sanity eval to see whether the route is worth expanding into a full Task038
matrix.

This is a diagnostic probe only. It cannot establish held-out morphology
generalization or TXL superiority.

## Minimal Closed Loop

Close this slice with:

- one selected held-out variant id;
- one checkpoint from `003` or `005`;
- train-variant eval JSON and held-out eval JSON under the same schema;
- failure classification if held-out quality drops.

## Evidence Gate

Evidence must record:

- held-out morphology id and split;
- checkpoint path;
- eval command;
- eval JSON path;
- train-vs-heldout metric comparison;
- explicit `full_matrix_claim=false` and `superiority_claim=false`.

## Subagent Ownership

Worker owns held-out sanity eval scripts/docs only. Worker must not expand to
multiple variants, speeds, or seeds unless the router opens a separate full
matrix task.

Reviewer checks that one held-out probe is not written as a generalization
claim.

## Failure Exit

If clean train-variant quality is not established, do not run held-out sanity.
If one held-out probe fails, record the failure and route to morphology/domain
randomization diagnosis instead of broadening the matrix.

## Log

- 2026-05-30 Opened as a gated diagnostic after clean quality evidence.
- 2026-05-30 Gated not run. Clean train-variant quality is not established for
  the current Task039 MLP or true-TXL route, and `004` routes to
  sequence-aware TXL PPO update before any held-out morphology probe. Per this
  slice's Failure Exit, held-out sanity eval should not be launched.

## Review

Status: gated not run; no held-out morphology evidence and no generalization or
superiority claim.
