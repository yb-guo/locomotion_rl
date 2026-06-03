# 006 Router Subagent Execution Plan

## Route

Make Task038 executable by router, worker subagents, and reviewers without
turning the planning task into implementation or training.

Execution order:

1. Router verifies scope, dirty worktree boundaries, and write scope.
2. Claim worker completes `001` fake evidence contract.
3. Slot worker completes `002` schema/mapping closed loop.
4. Morphology worker completes `003` local manifest, then requests H200 smoke
   only after slot review passes.
5. TXL worker completes `004` fake cache loop and then H200 runner smoke only
   after reset semantics are reviewable locally.
6. Eval worker completes `005` JSON evidence contract after `001-004` provide
   ids and metadata.
7. Reviewer validates all gates before any status is moved beyond pending.

Parallel boundaries:

- `001` and `002` may run in parallel because claim metrics and slot schema are
  independent.
- `003` local manifest may start after the slot schema draft exists, but H200
  load smoke must wait for `002` review.
- `004` fake cache work may run in parallel with `003`; H200 runner smoke must
  wait until inner/outer reset semantics are locally proven.
- `005` cannot claim comparability until `001` thresholds, `002` slot ids, `003`
  morphology ids, and `004` TXL metadata are available.

## Minimal Closed Loop

Close this slice by proving the router can assign every Task038 subtask without
opening implementation work:

- every subtask has an owner role;
- every subtask has a write scope;
- every subtask has a review gate;
- every subtask has a failure exit;
- parallel work is limited to independent local loops;
- H200 smokes are gated behind local evidence.

## Evidence Gate

Reviewable evidence is this file plus a header scan showing `task.md` and
`001-006` retain `Route / Log / Review`. The reviewer must also confirm that
the plan names no code edit, training run, asset download, checkpoint, or large
artifact as required work for this planning slice.

## Worker Write Scope

Router scope for this planning slice:

- allowed: `task.md`, `001-005` only for Minimal Closed Loop / Evidence Gate /
  Subagent Ownership clarifications, and this `006` file;
- disallowed: implementation code, training configs, checkpoints, generated
  assets, large logs/videos, `.test_tmp_task021/`, and unrelated task docs.

Future implementation workers must declare write scope before editing. Default
subtask write scopes:

- `001`: claim contract docs, small fake JSON/schema fixtures, local tests for
  aggregation if explicitly authorized.
- `002`: slot schema docs/files, G1 compatibility mapping, local schema tests.
- `003`: morphology manifest/config docs, small manifest fixtures, generator or
  H200 smoke script only if explicitly authorized.
- `004`: TXL cache contract docs, smallest policy/cache implementation and
  fake-env tests only if explicitly authorized.
- `005`: eval JSON schema/fixtures, eval runner wiring only if explicitly
  authorized.

## Subagent Ownership

- Router owns task sequencing, write-scope assignment, and stop/go decisions.
- Workers own only their declared subtask files and the smallest explicitly
  authorized evidence artifacts.
- Reviewers own gate checks and must refuse `passed` status when evidence is
  missing or the work escapes scope.

## Review Gates

Every subtask review must check:

- `Route / Log / Review` sections remain present;
- Minimal Closed Loop is local or H200-smoke sized, not a broad training run;
- Evidence Gate names an exact command output, JSON path, or schema fixture;
- Subagent Ownership names what the worker may and may not edit;
- failure exit is explicit and stops before unsupported claims;
- no pass status appears without evidence recorded in the subtask log.

Subtask-specific gates:

- `001`: claim is falsifiable and final-trial failure cannot be hidden by
  aggregate metrics.
- `002`: action dimension and slot meanings are stable across variants.
- `003`: variants are deterministic by seed, split-labeled, and slot-compatible;
  H200 load claims require JSON/log evidence.
- `004`: TXL memory preserve/clear semantics are proven locally before runner
  smoke; construction smoke is not adaptation evidence.
- `005`: all compared policies use the same variant ids, metrics, and eval
  matrix; skipped baselines are explicit.

## Failure Exit

Router stops the route and returns the blocker when:

- a worker needs to download assets, datasets, checkpoints, simulator assets, or
  upstream repos;
- a subtask requires changing topology or action dimension outside the fixed
  G1-like slot contract;
- Isaac Lab/Isaac Sim/RTX/Vulkan/Kit startup becomes the primary H200 blocker;
- evidence requires large training, large video/checkpoint output, or broad
  simulator runs not explicitly requested;
- another agent has modified the same files and the intended edit cannot be
  merged safely without clarification;
- a worker cannot produce the local fake/env/schema/JSON loop named by its
  subtask.

## Log

- 2026-05-29 Created router/subagent execution plan for Task038 planning slice.
  No implementation, training, asset download, or large output performed.
- 2026-05-29 Review subagent checked `task.md` and `001-005` for required
  `Route / Log / Review` sections, local closed loops, ownership, review gates,
  failure exits, and H200 gate ordering. No blocking findings.

## Review

Status: local planning closed with reviewer confirmation. Future H200 load
smoke, runner smoke, full eval, video, and claim execution remain gated by
`003`, `004`, and `005` evidence requirements.
