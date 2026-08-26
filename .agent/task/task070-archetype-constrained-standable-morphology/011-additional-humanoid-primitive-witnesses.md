# 011 — Additional Humanoid Primitive Witnesses

## Route

1. Parse the complete actuated source tree for X1 serial, X2 Ultra, T800,
   T800Pro, HU_D04, T1 23/29, and STAR1.
2. Preserve source joint order, parent/child body, type, normalized local axis,
   range, module, passive attachment bodies, head/hand branches, and exact
   source-to-anonymous motor accounting.
3. Emit anonymous box/capsule/cylinder geometry only. Do not copy source mesh,
   texture, logo, or model identity into the MJCF linkage.
4. Write a new attempt directory with descriptor, compilable XML, manifest,
   front/side/oblique/contact views, contact sheet, and execution-agent visual
   observations. Do not overwrite rejected or previously reviewed attempts.

## Log

- 2026-08-25：complete source-tree descriptors and anonymous primitive
  witnesses were generated for X1 serial `29`, X2 `31`, T800 `25`,
  T800Pro `43`, HU_D04 `31`, T1 `23/29`, and STAR1 `55`. X1's two
  config-only claw motors remain an explicit unresolved model/config gap rather
  than fabricated joints.
- Passive fixed head, sensor, palm, and finger branches are emitted as anonymous
  primitives while every actuated joint retains source order, parent/child
  body, local axis, range, and module. Candidate-only semantic slots are allowed
  only behind the fail-closed/no-policy-adapter metadata gate; the frozen 45-slot
  profiles are unchanged.
- Attempt006:
  `artifacts/preview_task070_v2_descriptor_driven_attempt006/`. It contains
  eight per-model XML/descriptor/manifest/four-view/contact-sheet bundles and
  `additional_humanoid_gallery.png`.
- The execution agent initially opened all eight sheets plus the aggregate gallery. The
  X1 right hip-pitch display nominal was mirrored to match its source coordinate
  convention so both support terminals reset at the same height; source axis,
  range, joint order, and 29-joint accounting were not changed.
- A later full-resolution recheck rejected the Booster T1 23/29 arm chains and
  invalidated the aggregate pass. Corrected attempt006 observation SHA-256:
  `1352dd423f7e470b571e0b8d627f34cf257401495f47d97f141aea291402cdcf`;
  aggregate gallery SHA-256:
  `29a46e660913478032a5878afe8751b7dd140cb1896105299f718be27fe749f6`.

## Review

- Candidate source-tree implementation: **execution-agent verified**. All eight
  exact source-to-anonymous motor bijections compile. Attempt006 visual status
  is **rejected** because T1 23/29 arm chains were collinear; the other six
  candidate sheets remain structurally readable. T1 remediation is tracked by
  microtask 013 and current attempt008.
- Focused Task070 pytest: `31 passed in 3.28s`; targeted Ruff:
  `All checks passed!`; frozen legacy/Task069 compatibility: `256/256`.
- The candidate extra-slot gate has negative coverage for missing fail-closed
  status, enabled policy adapter, pass-count opt-in, and mismatched declared
  extras; every case raises `ValueError`.
- Every artifact retains `user_visual_acceptance=false` and
  `counts_toward_task070_v2_pass=false`; the candidate witnesses do not
  promote a quantitative prior and do not pass Task070 v2.
