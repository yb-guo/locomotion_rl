# Subtask 001: Project Local XML Patch Generator

## Route

- Coding subagent owns the standalone patch generator and tests.
- Read-only reviewer reviews generated XML semantics before any H200 runtime
  ablation.
- Keep write scope small.

## Implementation Target

Add a standalone tool, likely:

```text
src/h200_locomotion_lab/tools/g1_ankle_roll_contact_patch.py
```

It should:

- accept `--source-asset` and default to the prepared G1 profile asset path;
- write patched XML variants under `outputs/task022/ankle_roll_contact_patch/assets/`;
- produce `summary.json` with source path, variant paths, changed target bodies,
  and exact geom/contact attr changes;
- never overwrite the source asset;
- support local fixture tests without Genesis.

Initial patch variants:

- `ankle_roll_friction_attrs`: add explicit `friction`, `condim`, and optional
  `priority` to the four existing ankle-roll support geoms.
- `ankle_roll_larger_spheres`: increase the four support geom sizes from
  `0.005` to a conservative larger value, while preserving positions.
- `ankle_roll_box_support`: replace or augment point-like support with a simple
  box-like support geom under each ankle-roll body.

## Stop Rules

- If patch semantics require changing body inertials, stop and create a separate
  subtask; do not mix inertia and contact geometry in subtask001.
- If XML parser cannot preserve enough structure to inspect the target geoms,
  stop before H200.

## Log

- 2026-05-13 Created with task022.

## Review

Status: pending.
