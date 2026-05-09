# 006: Review And Decision

## Goal

Decide whether task014 passed and whether task015 may start real learning
diagnosis/tuning.

## Route

1. Read all task014 logs and H200 artifacts.
2. Boundary scan:
   - no `GenesisG1SceneBackend` edits;
   - no SONIC/LocoFormer/ONNX/planner/render/GIF path in PPO smoke;
   - no downloads;
   - no `/mnt/workspace*` writes.
3. Correctness review:
   - GAE math;
   - tanh log-prob correction;
   - ratio/clip objective;
   - done/bootstrap handling;
   - device residency;
   - artifact integrity.
4. Evidence review:
   - local tests;
   - H200 tests;
   - H200 3-seed smoke;
   - metrics and checkpoint files.
5. Decision:
   - passed only if all acceptance criteria are met;
   - otherwise status remains failed/blocked with hypotheses and next action.

## Stop Rules

- Do not mark passed with missing H200 evidence.
- Do not mark passed with only 1 seed if plan requires 3.
- Do not mark passed if reward improves but diagnostics contain NaN/Inf.
- Do not mark passed if checkpoint exists but metrics are missing.

## Verification

- Review section in task.md updated.
- If passed, task014 can be merged.
- If failed, task014 stays open with diagnosis route.

## Log

- 2026-05-09 Reviewed task014 code, docs, and H200 artifacts.
- Boundary scan result:
  - no `GenesisG1SceneBackend` edit;
  - no LocoFormer, SONIC, ONNX, planner, render/GIF route in PPO smoke;
  - no download command or dependency fetch added;
  - no write/delete under `/mnt/workspace` or `/mnt/workspace1`.
- Correctness review result:
  - GAE masks done transitions;
  - tanh Gaussian log-prob clamps inverse action near bounds;
  - PPO ratio/clip objective is standard clipped surrogate;
  - value loss, entropy diagnostic, approx KL, clip fraction, and grad norm
    recorded per update;
  - tensor residency checked against `cuda:0`.
  - rollout diagnostic finite checks and terminal counters are batched to avoid
    per-step `.item()` synchronization.
- Evidence review result:
  - local focused tests passed;
  - local full pytest passed;
  - H200 focused tests passed;
  - H200 3-seed smoke passed;
  - metrics and final checkpoint exist.

## Review

Status: passed.

Decision: task014 passed. Task015 may start real learning diagnosis/tuning from
this PPO smoke loop, with the caveat that task014 does not claim walking
quality.
