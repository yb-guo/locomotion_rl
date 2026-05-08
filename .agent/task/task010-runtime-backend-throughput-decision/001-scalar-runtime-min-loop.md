# Route

Task: task010-runtime-backend-throughput-decision

Goal: add the smallest scalar runtime loop needed to prove backend-replaceable
single-robot execution semantics.

Scope:

- Add a scalar runtime module for one policy/control frame.
- Add an `ActionProvider` protocol or equivalent minimal boundary.
- Add fake/zero/sequence action providers only.
- Support `LogReplayG1RobotBackend` in local tests.
- Keep the runtime out of vectorized PPO hot paths.

Environment:

- local Windows workspace for code and unit tests
- no simulator dependency at import time
- no remote/H200 command required for this subtask

Verify:

- local focused tests for replay runtime pass
- full local suite passes or failure is recorded with exact cause

No Hack:

- Do not hard-code task fixture paths into runtime code.
- Do not import Genesis or ONNX at module import time.
- Do not route PPO/vectorized training through scalar runtime.

Hardware: local only.

# Log

- 2026-05-08 local: Added `h200_locomotion_lab.runtime.scalar_g1_runtime`
  with `ActionProvider`, `ZeroActionProvider`, `FakeActionProvider`,
  `SequenceActionProvider`, and `ScalarG1Runtime`.
- Runtime uses existing `G1RobotBackend`, `G1RobotState`, `G1MotorCommand`,
  and `LogReplayG1RobotBackend`; it does not import Genesis or ONNX and does
  not route vectorized training through the scalar loop.
- Focused verification:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_genesis_g1_throughput_probe.py tests/test_scalar_g1_runtime.py`
  -> 10 passed, 1 pytest cache warning from Windows `.pytest_cache` access.
- Full local verification:
  `$env:PYTHONPATH='src'; python -m pytest`
  -> 119 passed, 1 pytest cache warning from Windows `.pytest_cache` access.
- Static lint attempt:
  `python -m ruff check ...` -> not run because `ruff` is not installed in
  the local interpreter.

# Review

Status: local implementation passed. H200 evidence is not required for this
subtask.
