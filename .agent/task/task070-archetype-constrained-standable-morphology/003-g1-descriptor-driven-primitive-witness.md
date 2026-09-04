# 003 — G1 Descriptor-Driven Primitive Witness

## Route

1. Generate one anonymous primitive G1 witness from the descriptor produced by
   `002`, using source body-tree offsets normalized into a clean local frame.
2. Replace visual/collision mesh with boxes/capsules only; do not copy vendor
   mesh, texture, logo, body names, or exact visual identity.
3. Preserve all 29 actuated joints in order and emit manifest fields for source
   motor count, anonymous motor count, parent-child edge coverage, local axes,
   and module DoF sums.
4. Render front/side/oblique/contact sheet with visible joint markers or labels
   sufficient to inspect shoulder, elbow, wrist, waist, hip, knee, and ankle
   chains.

## Log

- Completed 2026-08-25. Replaced the earlier hand-authored v2 preview path
  with a descriptor-driven builder: `MotorDofPreservingArchetypePreviewGenerator`
  now calls `load_g1_motor_dof_preserving_descriptor()` and instantiates
  anonymous primitive links from parsed source-tree parent/child edges.
- The preview uses anonymous link names, primitive box/capsule geometry, and
  joint marker sites. It does not copy source mesh, texture, Logo, body names,
  or model identity into the generated MJCF.
- Generated new preview artifacts under
  `artifacts/preview_task070_v2_descriptor_driven/unitree_g1_seed000/`:
  XML `526ed69350067fd329f4ed9fa3fed03d3d0879e7dc6345607c8c0730bdad1de3`,
  manifest `6e62f9899baf3f76de883602e7b08c475e4de94c7460113ac8cbe82330fa1906`,
  sheet PNG `967e8d2cd031c0f7b525cfa06966f3b329f9a34c95f9aba08857e3da1bfe56d2`.
- Superseded after LocoFormer Figure 6 visual comparison. Although the source
  descriptor and motor accounting were correct, this first descriptor-driven
  render used grey/default vertical capsule visuals and was rejected as not
  visually acceptable.
- Generated replacement attempt002 under
  `artifacts/preview_task070_v2_descriptor_driven_attempt002/unitree_g1_seed000/`
  with v2-only colored primitive modules and directed capsule edges:
  XML `d4d9c0c5e35913bc58542ce4c334d385fbd5e0e6c3e56b90a602ace87a1f972e`,
  manifest `5e207ddb966a63b117d1d4347e9873ed7b65b12003382ac3086ceff52700b805`,
  sheet PNG `6d566d9667a36d1658b2c1edf7eee14cf72c59358209a5ebc15c3737973f5e85`.
- Manifest reports `source_actuated_motor_count=29`,
  `anonymous_motor_count=29`, `total_actuator_count=29`,
  `model_nu=29`, `module_dof_counts=6+6+3+7+7`, and
  `all_selected_motor_edges_preserved=true`.
- Rendered front/side/oblique/contact sheet with joint marker sites. Stance was
  not run; `stance_claim=not_run_preview_only`.
- Verification:
  `.venv/bin/python -m h200_locomotion_lab.tools.task070_morphology_verification preview-v2 --output-dir .agent/task/task070-archetype-constrained-standable-morphology/artifacts/preview_task070_v2_descriptor_driven/unitree_g1_seed000 --seed 0`
  completed.
- Focused pytest and Ruff checks passed as recorded in `002`.
- User rejected attempt002's link structure. Attempt003 fixes the reproduced
  causes rather than recoloring it: the descriptor now records source
  body-local quaternions, the v2 compiler emits those quaternions on anonymous
  bodies, each directed capsule ends at the next descriptor joint origin, and
  v2-only local geom/attachment metadata separates torso, waist, pelvis,
  shoulder/hip connectors, ankle terminal, and footpad. The elbow audit pose is
  recorded as preview-only and does not change joint tree, axis, range, or motor
  accounting.
- Final attempt003 artifacts under
  `artifacts/preview_task070_v2_descriptor_driven_attempt003/unitree_g1_seed000/`:
  descriptor `d7933388ee454ed4fb3f76a7f9b52859637104ea7fb93e1669422ee75a844c26`,
  XML `6ede5976ea1135bd049e1148d223d5156aedd67f21f9fa97174b8514e83b77f1`,
  sheet `3ef629e63db83caa6d0e83f6165aae0ef9658e66a3c6e75456669c1b2050f070`,
  manifest `111470c4b2ce492adb513ed75806749b37f0c8385b99dcc7ef6022b4b7760bd5`,
  visual observation `dcd96fecec5b1414cc2b103dc4e588800b1b601dc85d7bd18f18f7849bef95bb`.
- Attempt003 manifest/MuJoCo remain `29 -> 29`, module sums `6+6+3+7+7`,
  `model_nu=29`, primitive-only, no mesh/texture/Logo, and
  `stance_claim=not_run_preview_only`.
- Verification: full `tests/test_task070_morphology.py` is `15 passed in 1.74s`;
  requested four-file Ruff is `All checks passed!`; explicit legacy/Task069
  compatibility is `256/256` passed.

## Review

- Attempt002 is rejected by the user and must not be used as user-accepted
  visual evidence. Its 29-motor accounting is intact, but its rendered link
  hierarchy is structurally misleading: outgoing capsules inherit incoming
  edge length, source body-local quaternion frames are dropped, and the torso
  box overlaps the pelvis/waist stack.
- Attempt003 passes the execution-agent visual-inspection gate only. It remains
  pending user acceptance and does not count toward Task070 v2 pass evidence.
- Fail if the witness again looks like boxes plus collapsed rods, if arm/leg
  segments are visually merged, or if geometry is not descriptor-driven.
