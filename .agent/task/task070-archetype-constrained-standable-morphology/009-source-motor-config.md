# 009 — Source Motor Config Priors

## Route

1. Inventory local motor/actuator configuration evidence for G1, PM01, Spot,
   Go2, Lite3, and the local terminal-wheel module without downloading assets.
2. Preserve source-declared values and provenance in each motor descriptor,
   while rejecting obvious placeholders instead of silently treating them as
   physical truth.
3. Derive anonymous final effort/PD/armature values from trusted hints and the
   realized primitive-link lever scale; keep raw/proxy and final values separate.
4. Regenerate a new attempt004 and verify frozen legacy/Task069/Task070-v1
   outputs remain stable.

## Log

- 2026-08-25: local evidence inventory:

  | Center | usable local config | important classes | confidence |
  | --- | --- | --- | --- |
  | G1 | `g1_constants.py` | 7520-22 `139/20`, 7520-14 `88/32`, parallel 5020×2 `50/37`, 5020 `25/37`, 4010 `5/22` (effort/velocity) plus PD/armature | upstream companion config; hash-gated |
  | PM01 | selected URDF limits | high-torque `164/26.3`; standard `52/35.2` | limit-only, no PD/armature |
  | Go2 | `go2_constants.py` | hip/thigh `kp=20,kd=1,effort=23.5,armature=.01`; calf `40/2/45/.02` | upstream companion config; hash-gated |
  | Lite3 | selected URDF limits | hip `24/26.2`; knee `36/17.3` | limit-only, no PD/armature |
  | Spot | selected URDF declares uniform `1000/1000` | rejected as placeholder for all 12 joints | raw evidence retained, not consumed |
  | terminal wheels | no authorized source config | local torque limit `45`, joint damping `.15`, armature `.01` | explicit local engineering module |

- Added `SourceMotorConfig` to the Task070 v2 descriptor. Every selected source
  motor now records raw declarations, source path/SHA, motor class, confidence,
  usability, and any rejection reason. G1 and Go2 hard-coded audited tables fail
  closed if their companion-config SHA changes.
- Anonymous final configs use trusted effort and PD/armature when available.
  URDF limit-only centers derive a documented generic PD proxy from effort
  class. Final effort scales with realized lever length; gains and armature
  scale with lever squared. Spot uses an anonymous family fallback because its
  placeholder values are rejected. Source velocity remains a controller-side
  metadata hint and is not falsely reported as enforced by a MuJoCo position
  actuator.
- LocoFormer public boundary was checked against the
  [official paper](https://arxiv.org/html/2509.23745v1) and
  [project page](https://generalist-locomotion.github.io/): the policy emits
  target joint positions in a unified joint space; procedural robots avoid
  exact market parameters and broadly randomize control gains, joint limits,
  mass, center of mass, and standard dynamics. The public material does not
  provide an exact motor table, randomization ranges, or an official generator
  implementation. Task070 therefore uses real configs only as audited prior
  hints and keeps exact named-robot parity false.
- attempt004 contains 10 regenerated witnesses (all five centers, non-wheel and
  wheeled), each with XML, structural descriptor, manifest, four views, and a
  contact sheet. The execution agent reopened all 10 sheets; regenerated image
  SHA values were unchanged after the final source hash gate. All manifests
  retain `user_visual_acceptance=false` and
  `counts_toward_task070_v2_pass=false`.
- Evidence:
  - `artifacts/preview_task070_v2_descriptor_driven_attempt004/motor_config_audit.json`
  - `artifacts/preview_task070_v2_descriptor_driven_attempt004/motor_config_agent_visual_observation.json`
  - `artifacts/preview_task070_v2_descriptor_driven_attempt004/validation_summary.json`
- Validation:
  - focused pytest: `21 passed in 2.73s`;
  - required four-file Ruff: `All checks passed`;
  - frozen compatibility: `256/256 passed`;
  - attempt004 manifest/anonymity/config audit: `PASS` for 10/10 manifests.

## Review

- Scoped motor-config integration: **PASS**. Real available config distinctions
  are no longer flattened, raw and final values are auditable, Spot placeholders
  fail closed, and wheels are not mislabeled as vendor-derived.
- Task070 v2 overall: **NOT PASSED**. This microtask does not establish the full
  sampler, stance matrix, training distribution, policy behavior, sim2real, or
  user visual acceptance.
