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

## Review

Status: pending.
