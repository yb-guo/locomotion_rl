# 002 — G1 Source Tree Parser

## Route

1. Parse the local pinned G1 MJCF only; do not download new assets.
2. Extract every selected actuated motor joint with source joint name, parent
   body, child body, body-local `pos`, normalized joint axis, range, and semantic
   module.
3. Emit a G1 v2 structural descriptor artifact with exact 29 source motors and
   source-to-anonymous slot mapping.
4. Add focused tests for 29 unique motors, waist/arm/leg branch coverage, no
   omitted wrist motors, and source body-tree edge preservation.

## Log

- Completed 2026-08-25. Added a parsed G1 motor-DoF-preserving descriptor path
  in `src/h200_locomotion_lab/robots/archetype_morphology.py`:
  `load_g1_motor_dof_preserving_descriptor()` recursively parses the pinned
  local G1 MJCF only.
- Extracted per selected motor: source joint name, joint type, source parent
  body, source child body, body-local `pos`, joint-local `pos`, normalized
  local axis, range, module, source tree depth, and anonymous parent/child link.
- Emitted descriptor artifact:
  `artifacts/preview_task070_v2_descriptor_driven/unitree_g1_seed000/unitree_g1_29dof_structural_descriptor.json`
  SHA `e0b44aad94001ba7252fd00d1d2229a46f7e6023754f6c7e388eff434251019a`.
- Added focused tests for 29 unique motors, module DoF counts
  `left_leg=6/right_leg=6/waist=3/left_arm=7/right_arm=7`, wrist/waist
  coverage, source body-tree edge preservation, axis normalization, and
  descriptor-driven anonymous edge preservation.
- Verification:
  `.venv/bin/python -m pytest -q tests/test_task070_morphology.py::test_task070_v2_g1_descriptor_preserves_source_body_tree_edges tests/test_task070_morphology.py::test_task070_v2_g1_preview_edges_are_descriptor_driven tests/test_task070_morphology.py::test_task070_v2_g1_preview_preserves_29_motor_dofs_and_compiles`
  passed.
- Ruff passed for touched Task070 files.

## Review

- Pass. G1 `29 -> 29` accounting is derived from parsed source tree
  fields, not a hand-written geometry table.
- Fail on duplicate source motors, missing wrist/waist motors, generic
  `semantic=other`, disconnected parent-child edges, or axis/range omissions.
