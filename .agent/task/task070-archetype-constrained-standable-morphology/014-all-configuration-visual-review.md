# 014 — All-Configuration Torso/Hub Fix and User Review Bundle

## Route

1. Reopen attempt008 and the eight additional humanoid witnesses at full
   resolution; treat readable arm chains as insufficient if torso/pelvis hub
   geometry overlaps or detaches.
2. Preserve every parsed source body position/quaternion and every selected
   joint parent/child, order, axis, range, module, and actuator count. Change
   only anonymous candidate box geometry center/size.
3. Pair the descriptor root with the nearest branching waist hub and derive a
   small target surface gap from their accumulated source attachment pose.
4. Generate a new, non-overwriting attempt containing the current 10 base/
   wheel configurations and eight additional humanoid candidates, then open all
   four-view sheets before setting the execution-agent visual flag.
5. Put all 18 configurations into the local MuJoCo flat arena and keep actuator
   response, stance, and walking claims separate.

## Log

- The rejected layout placed every candidate root box at a fixed
  `+0.08 * scale` world-Z offset and every paired hub at a fixed
  `+/-0.17 * scale` offset. It removed intersections but produced nominal
  torso/pelvis surface gaps of `0.2067 m` for both Booster T1 variants and
  `0.1926 m` for STAR1: a visibly detached waist. Attempt008 also retained the
  earlier zero-centered torso/pelvis overlap and is now marked rejected even
  though its T1 arm audit pose remains useful.
- The candidate-only fix computes
  `S = max(0, root_half_z + hub_half_z + target_gap - abs(d_z))`, where `d_z`
  is the accumulated source attachment world-Z distance after excluding the
  source root translation that the generated free root replaces. The root and
  paired hub move outward by `0.4S` and `0.6S`; cumulative source quaternions
  convert the world-Z displacement to body-local geom centers. The target gap
  is `0.02 * visual_scale`. Other hubs are not displaced.
- Final nominal MuJoCo surface gaps are `0.0100569121 m` for X1 and
  `0.0201138258–0.0201138330 m` for X2, T800, T800Pro, HU_D04, T1 23/29, and
  STAR1. The regression gate requires every gap to remain strictly between
  `0.008 m` and `0.060 m`.
- Unified visual artifacts are in
  `artifacts/preview_task070_v2_descriptor_driven_attempt009/`: 18 XML files,
  18 structural descriptors, 18 manifests, 72 individual frames, 18 contact
  sheets, and three labeled review galleries. The execution agent opened all
  18 sheets; the eight candidate formal sheets are byte-identical to the
  separately opened post-fix preflight sheets.
- Aggregate observation:
  `all_configuration_agent_visual_observation.json`, SHA-256
  `922ec777fcbd7a21609d73c7d8964d9dccf7eb1e0452c198426d8822cd0dce46`.
  All-configuration gallery SHA-256:
  `0f8ebc881fbfa19861f28c8065b3e5e813cc2a9b75e70df3bc2c1a0da330d513`.
- Flat-arena artifacts are in `artifacts/arena_task070_v2_attempt009/`.
  Evidence SHA-256:
  `d9dcc45142026043aa23532f14324e23fb24dd03c084f59f0d2dfb1296af23eb`;
  gallery SHA-256:
  `027c140312dd704f3f3a4e4004e2674e2adf007396846e7f43db548c2c4dfe30`.
  Results are `18/18` compile, exact accounting, reset, and paired-baseline
  all-actuator response; generic stance is `0/18` and `walking_claimed=false`.
- Validation: focused Task070 pytest `31 passed in 3.31s`; targeted Ruff
  `All checks passed!`; frozen legacy/Task069 compatibility `256/256`.

## Review

- Attempt008 complete T1 witness: **rejected after torso/pelvis overlap
  recheck**. Its shoulder/elbow/wrist pose is retained in the corrected build.
- Attempt009 18-configuration visual bundle: **execution-agent visual check
  passed**. Torso/pelvis or trunk/limb hierarchy, attachments, segmented chains,
  and terminal feet/wheels are readable without overlapping or detached body
  boxes.
- Every attempt009 manifest and aggregate observation retains
  `user_visual_acceptance=false` and
  `counts_toward_task070_v2_pass=false`. Task070 is not passed; candidate
  transmission/motor evidence remains fail closed, and no standing or walking
  claim is made.
- Independent read-only review reported no P0/P1/P2 findings. It confirmed
  candidate-only geometry scope, byte-identical frozen descriptors/XML, all 18
  image/manifest bindings, anonymous XML identity, and the same stance/walking
  claim boundary.
