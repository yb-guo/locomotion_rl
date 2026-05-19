# Subtask 001: Fixed Controller Probe

## Route

- Coding subagent owns the standalone probe and local tests.
- Keep write scope small.
- Read-only reviewer reviews controller semantics before H200.

## Implementation Target

Add a standalone tool, likely:

```text
src/h200_locomotion_lab/tools/g1_base_stabilizer_probe.py
```

The probe should:

- accept `--asset-path`;
- accept stabilizer mode: `none`, `attitude`, `height`, `attitude_height`;
- keep controller output bounded and report clipping/saturation;
- report first tilt/reset, root height/upright, joint errors, and contact
  metrics when available;
- run local tests without importing Genesis.

## Stop Rules

- If stabilizer design requires source asset or inertial edits, stop.
- If the local seam cannot test command construction and controller math, stop
  before H200.

## Log

- 2026-05-13 Created with task023.
- 2026-05-13 Coding subagent added a standalone local deterministic probe at
  `src/h200_locomotion_lab/tools/g1_base_attitude_height_stabilization.py`.
  The default runner is pure Python (`local_toy`) and imports no Genesis backend
  at module import time. The reserved `genesis` runner entry delays backend
  import until the Genesis runner function is explicitly invoked.
- 2026-05-13 Added focused local tests at
  `tests/test_g1_base_attitude_height_stabilization.py` for CLI modes and asset
  metadata, gain/output clipping, summary schema, first tilt/reset detection,
  top joint/contact summaries, local improvement classification, JSON artifact
  output, and guarded H200 Genesis command construction.
- 2026-05-13 Verification evidence:
  `PYTHONPATH=src python -m pytest tests/test_g1_base_attitude_height_stabilization.py -p no:cacheprovider`
  passed locally with 7 tests.

## Review

Status: reviewed_no_blocking.

- 2026-05-13 Read-only reviewer found no blocking findings for subtask001 local
  readiness. Reviewer confirmed the standalone local feedback loop, controller
  clipping, reset/contact/top-joint summaries, delayed Genesis import, and stop
  rule compliance.
- Non-blocking carry-forward for subtask002: implement the real `genesis`
  runner before the H200 matrix and ensure generated guarded commands set
  `CUDA_VISIBLE_DEVICES=1`.
