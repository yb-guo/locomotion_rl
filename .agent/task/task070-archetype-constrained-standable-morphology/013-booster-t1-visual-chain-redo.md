# 013 — Booster T1 Visual-Chain Redo

## Route

1. Freeze the full-resolution rejection of the attempt006 Booster T1 23/29
   sheets and retain attempt007 as a rejected partial fix.
2. Preserve the parsed T1 source body tree, joint order, parent/child edge,
   local axis, range, module, and exact 23/29 actuator accounting.
3. Use only real T1 generalized joints and in-range visual-audit nominals to
   rotate the collinear arm chain into a readable pose. Do not alter source
   offsets or add synthetic links.
4. Generate a new attempt with XML, descriptor, manifest,
   front/side/oblique/contact views, sheets, and execution-agent observations.
5. Re-run compile/reset/paired-baseline actuator smoke separately from stance
   and walking claims.

## Log

- 2026-08-25：full-resolution recheck found that attempt006 T1 29 rendered six
  arm DoFs as an almost single lateral rod; T1 23 had the same issue more
  mildly. Attempt006 aggregate visual status was corrected to false.
- Attempt007 added mirrored shoulder-roll `-0.42/+0.42 rad`. It separated the
  upper arm and forearm, but T1 29 wrist/hand remained collinear; the execution
  agent retained it as rejected instead of overwriting it. Observation
  SHA-256:
  `a569fc0764ef4fcf122eb880bbd3e3cc9d25d39b1f2b577d193747a306eb35cb`.
- Attempt008 additionally uses the real mirrored elbow-yaw joints at
  `+0.58/-0.58 rad`. MuJoCo forward positions place the downstream wrist
  chain at distinct lateral, forward, and vertical coordinates while all source
  descriptor fields and 23/29 motor counts remain unchanged.
- Current artifacts:
  `artifacts/preview_task070_v2_descriptor_driven_attempt008/`. The execution
  agent opened both sheets and every front/side/oblique/contact image.
  Observation SHA-256:
  `7acb7838bac9d39da1522af5856a4e4e8fa92d02421dbf170404b66185901779`.
- T1-only flat-arena evidence:
  `artifacts/arena_task070_v2_attempt008/t1_flat_arena_smoke.json`, SHA-256
  `86248d19063e750cf8d842b7488ebcabd69db93299f8bb0ae2a16d4db326d95f`.
  Both models compile, preserve exact accounting, reset, and respond on every
  actuator under the paired-baseline contract. Generic stance remains `0/2`;
  `walking_claimed=false`.
- Validation: focused Task070 pytest `31 passed in 3.28s`; targeted Ruff
  `All checks passed!`; frozen legacy/Task069 compatibility `256/256`.
- 2026-08-26：full-witness recheck found that attempt008 still used the old
  zero-centered root/waist boxes, so its torso and pelvis materially overlapped.
  The arm-chain pose remains valid, but attempt008 is rejected as a complete
  visual witness and superseded by the source-attachment-gap layout in
  attempt009. Both attempt008 manifests and its observation now record
  `agent_visual_check_passed=false`.

## Review

- Attempt006 T1 visual result: **rejected**.
- Attempt007 partial visual fix: **rejected**.
- Attempt008 T1 23/29 arm subchain: readable, but the complete witness is
  **rejected after torso/pelvis overlap recheck**.
- `user_visual_acceptance=false` and
  `counts_toward_task070_v2_pass=false`; no candidate prior promotion, stance,
  walking, or Task070 pass is claimed.
