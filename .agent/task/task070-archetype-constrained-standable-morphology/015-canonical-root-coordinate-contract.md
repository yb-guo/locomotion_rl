# 015 — Canonical Root Coordinate Contract

## Route

1. Audit the native free-root body and local axis convention for every current
   Task070 v2 structural center without changing its parsed source tree.
2. Define one virtual, right-handed `canonical_root`: +X forward, +Y left,
   +Z up; biped origin at the first left/right hip attachment midpoint and
   quadruped origin at the four first-hip attachment centroid.
3. Emit the frame as a transparent MJCF site and versioned manifest transform,
   then expose one public runtime reader for pose, local twist, and projected
   gravity.
4. Route WholeBody observation, reward, upright, and fall-height consumers
   through the canonical state while preserving exact no-site legacy fallback.
5. Bind all 18 configurations to a new, non-overwriting attempt, run geometric
   and runtime regressions, open the review galleries, and keep user acceptance
   and Task070 pass flags false.

## Log

- Native source roots are not semantically uniform: most humanoids root at a
  pelvis/base, Booster T1 roots at `Trunk` above an actuated waist, STAR1 roots
  above its waist chain, and quadrupeds root at a trunk. Physically rerooting
  T1/STAR1 would reverse source parent/child edges and violate the v2 descriptor
  contract, so the solution is a virtual site rather than a rewritten tree.
- `canonical_root_frame_v1` stores both transform directions, `wxyz` ordering,
  explicit right-handed axes, pre-global-scale translation units, and the fact
  that native free-root qpos is not canonical. The v2 embodiment contract and
  hash now include `canonical_root_v1`; frozen legacy, paper-faithful, and v1
  contract identities are unchanged.
- Runtime inspection found and fixed three fail-open gaps: additional humanoid
  candidates initially missed the site, projected gravity used the wrong
  rotation-matrix row, and the stance solver inferred leg joints from generated
  link names instead of stable `limb*` semantic slots.
- Attempt010 artifacts:
  `artifacts/preview_task070_v2_descriptor_driven_attempt010/`. The audit covers
  the exact ten base/wheel cases plus all eight additional humanoid candidates:
  `18/18` passed, maximum hip-centroid residual
  `4.999999969612645e-09 m`, maximum orthogonality error `0`, and minimum
  +X/+Y/+Z alignment `1.0`. T1 and STAR1 also pass a dynamic-waist probe at
  `q=0.31 rad`, `qvel=0.8 rad/s`: current hip centroid, parent orientation,
  local twist, and projected gravity all have zero recorded residual. Audit
  SHA-256:
  `f96da04079f8155221b4067cac6af31968182209f809a08ee1face32d28b8547`.
- The execution agent opened the all-configuration, humanoid, quadruped/wheel,
  and full-resolution G1/T1/Spot sheets. Primitive linkage is unchanged from
  the accepted execution-agent attempt009 geometry; the runtime site is
  transparent. Visual observation SHA-256:
  `0251b4b8e3cbcdf7984676a659669a8860c6d13664f846c8ff06c307451c141a`.
- Flat-arena attempt010 remains an actuator smoke only: `18/18` compile,
  accounting, reset, and all-actuator paired response; stance `0/18` and
  `walking_claimed=false`. Evidence SHA-256:
  `8a8d281d9ede3d32713ceb92c96976f8eacab864d35d6feba1c31fb2db52436d`.
- Validation: Task070 focused pytest `35 passed`; WholeBody contract/extended
  pytest `30 passed`; targeted Ruff passed. A real G1 v2 WholeBody shard also
  initialized with the static stance solver after semantic leg-chain repair.

## Review

- 2026-08-26 append-only user acceptance overlay：用户明确表示“目前我认证过了，感觉还行”；`user_visual_acceptance=true`，绑定 audit `f96da04079f8155221b4067cac6af31968182209f809a08ee1face32d28b8547`、visual `0251b4b8e3cbcdf7984676a659669a8860c6d13664f846c8ff06c307451c141a`、arena `8a8d281d9ede3d32713ceb92c96976f8eacab864d35d6feba1c31fb2db52436d`。`counts_toward_task070_v2_pass=false`、stance `0/18`、walking `false`、Task070 not passed 保持不变。

- Canonical-root geometric/runtime audit: **execution passed**.
- Attempt010 primitive-link visual check: **execution agent passed**.
- Follow-up high-risk read-only review: **no P0/P1/P2 findings**. Its initial
  T1/STAR1 dynamic-waist test-gap finding was resolved by the two-case
  `qpos=0.31`, `qvel=0.8` regression and rebound audit.
- Frozen attempt010 JSON still records `user_visual_acceptance=false`; the
  append-only overlay above is now the effective `user_visual_acceptance=true`
  record. `counts_toward_task070_v2_pass=false` remains mandatory.
- This work does not turn the flat-arena smoke into a stance or walking claim;
  Task070 remains not passed.
