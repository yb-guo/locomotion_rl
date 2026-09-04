# 004 — G1 Preview Visual Review

## Route

1. Open the `003` preview sheet locally and inspect every view.
2. Record a visual observation manifest with family, seed, view, visible module
   coverage, attachment sanity, terminal contact geometry, clipping, overlaps,
   and problems observed.
3. Set `agent_visual_check_passed=true` only after the execution agent's local
   image-viewer check confirms the preview is structurally inspectable.
4. Stop after agent visual pass and present the preview sheet, manifest, and any
   residual visual concerns to the user for human acceptance.
5. Record `user_visual_acceptance=true` only after the user explicitly confirms
   the preview is acceptable.
6. Keep this as a preview-quality gate only; do not claim stance, full R4, or
   policy behavior.

## Log

- Agent visual check completed 2026-08-25 using the local image viewer on
  `artifacts/preview_task070_v2_descriptor_driven/unitree_g1_seed000/unitree_g1_29dof_anonymous_preview_sheet.png`.
- Wrote visual observation artifact
  `artifacts/preview_task070_v2_descriptor_driven/unitree_g1_seed000/unitree_g1_29dof_agent_visual_observation.json`
  SHA `31c33a0dff6cef410fda30fc5c63920c30f8571f8509491ee614a13c37ce8dd1`.
- Recorded `agent_visual_check_passed=true` and
  `user_visual_acceptance=false`; preview manifest status is now
  `descriptor_driven_preview_agent_visual_check_passed_pending_user_acceptance`.
- Visual observations: front view shows torso/pelvis, both legs, feet, and
  side shoulder/wrist markers, though arm chains are partly occluded by torso;
  side and oblique views expose leg chain, foot contact boxes, arm chain, and
  wrist markers; contact view exposes terminal foot geometry. No critical
  clipping, collapsed chain, or incoherent overlap observed.
- Residual visual limitation: waist joints are compact and best audited from
  manifest plus side/oblique marker positions, not front pixels alone.
- User challenged this visual against LocoFormer Figure 6, and the challenge
  was accepted. The `unitree_g1_seed000` descriptor-driven attempt was revoked:
  its manifest now has `agent_visual_check_passed=false` and status
  `descriptor_driven_preview_rejected_after_locoformer_figure6_visual_comparison`.
  Failure modes: grey torso/vertical capsule visual language, insufficient
  LocoFormer-style colored primitive-link readability, and capsule geometry not
  drawn along descriptor edge directions.
- Reworked the v2 preview render path without changing legacy profile
  dataclasses: `compile_mjcf()` now honors v2 metadata
  `capsule_local_fromto` and `link_visual_rgba`; the G1 descriptor builder emits
  directed capsule vectors and high-contrast primitive-link colors.
- Generated attempt002 under
  `artifacts/preview_task070_v2_descriptor_driven_attempt002/unitree_g1_seed000/`.
  Artifact SHA-256: descriptor
  `e0b44aad94001ba7252fd00d1d2229a46f7e6023754f6c7e388eff434251019a`;
  XML `d4d9c0c5e35913bc58542ce4c334d385fbd5e0e6c3e56b90a602ace87a1f972e`;
  sheet PNG `6d566d9667a36d1658b2c1edf7eee14cf72c59358209a5ebc15c3737973f5e85`;
  manifest `5e207ddb966a63b117d1d4347e9873ed7b65b12003382ac3086ceff52700b805`;
  agent visual observation
  `292230ac4173702f76c3d56d4e6b421bf0c23b8c9e2b73680f4f798b499b5261`.
- Attempt002 local image-viewer check passed against LocoFormer Figure 6
  visual criteria: colored primitive modules, directed limb segments, visible
  terminal feet, and visible full 29DoF G1 arms/wrists as an anonymous witness.
  It is still not an official LocoFormer generator reproduction and not stance
  evidence.
- Stopped here for user human visual acceptance on attempt002. This preview
  still has `counts_toward_task070_v2_pass=false` until explicit user
  acceptance.
- 2026-08-25 user rejection supersedes the attempt002 execution-agent visual
  verdict. Local re-inspection froze these structural failures before attempt003:
  - the blue torso box is centered on the waist-pitch origin and extends down
    over the red pelvis, so pelvis, three-axis waist, and torso do not read as
    separate load-bearing modules;
  - `_g1_link_visual_length()` mixes the incoming body offset into an outgoing
    capsule. For example the roughly 1.8 cm ankle-pitch-to-ankle-roll edge is
    rendered at roughly 31 cm, and the roughly 4 cm shoulder-pitch-to-roll edge
    is rendered at roughly 27 cm;
  - the parser records body-local positions but drops source body quaternions,
    so downstream segment directions and local joint axes are instantiated in
    the wrong anonymous frames;
  - the front view consequently collapses the shoulder/elbow/wrist hierarchy
    into a bar plus short colored rods, while long duplicated ankle geometry
    hides the knee/ankle/foot transition;
  - the footpad is positioned from the duplicated terminal capsule length
    instead of the terminal ankle endpoint, weakening the visible ankle-to-foot
    attachment.
- Attempt002 remains a rejected immutable artifact with
  `counts_toward_task070_v2_pass=false`; attempt003 must be written to a new
  directory and independently re-opened in the local image viewer.
- Final attempt003 contact sheet and all four individual PNGs were opened with
  the local image viewer after the final regeneration. Front shows separate
  torso/waist/pelvis, both shoulder and hip connectors, both downward arm/wrist
  chains, segmented legs, ankles, and feet. Side exposes the elbow audit pose,
  knee/ankle bend, and foot length; oblique separates mirrored branches; contact
  shows both terminal foot boxes against the floor. No critical clipping or
  incoherent structural overlap was observed.
- The official Figure 6 biped and wheeled-biped images were inspected for their
  narrow visual language: box trunk, compact attachment, directed colored limb
  segments, and terminal feet/wheels. Attempt003 matches that primitive-link
  readability direction without claiming official generator, parameter, or
  pixel parity.
- Wrote
  `artifacts/preview_task070_v2_descriptor_driven_attempt003/unitree_g1_seed000/unitree_g1_29dof_agent_visual_observation.json`
  SHA `dcd96fecec5b1414cc2b103dc4e588800b1b601dc85d7bd18f18f7849bef95bb`.
  Recorded `agent_visual_check_passed=true`, while
  `user_visual_acceptance=false`, `counts_toward_task070_v2_pass=false`, and
  `stance_claim=not_run_preview_only` remain fail-closed.

## Review

- Attempt002 failed user visual review. Attempt003 passed the execution-agent
  visual gate and is pending explicit user acceptance; it is not Task070 pass
  evidence.
- Fail if any critical chain is hidden, clipped, collapsed, or ambiguous enough
  that a reviewer cannot audit motor preservation from the image plus manifest.
- Fail if `agent_visual_check_passed=true` is missing, if the agent skipped local
  image viewing, or if `user_visual_acceptance=true` is missing.
