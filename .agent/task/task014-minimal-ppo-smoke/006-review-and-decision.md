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

Pending implementation.

## Review

Status: pending.
