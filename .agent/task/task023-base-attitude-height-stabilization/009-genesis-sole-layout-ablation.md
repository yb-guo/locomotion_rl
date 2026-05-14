# Subtask 009: Genesis Sole Layout Ablation

## Route

- Continue inside task023; no new top-level task.
- Genesis-only. No MuJoCo, PPO, downloads, `/mnt/workspace*`
  writes/deletes, or `GenesisG1SceneBackend` changes.
- Do not promote `ankle_roll_box_support` as final asset. Use it as a
  reference to isolate which support semantics matter.

## Feedback Loop

```text
mesh/static audit -> generate reviewed sole-layout candidates ->
40-step early-force gate -> 140-step passive/active horizon gate
```

## Ranked Hypotheses

1. **Subtask008 failed because disabling point supports removed heel/toe
   support semantics.**
   - Prediction: a center sole box that keeps source point supports preserves
     box-support-like passive horizon, while the same box with points disabled
     regresses toward source/subtask008.
2. **The support-sphere footprint was not the right continuous sole layout.**
   - Prediction: a mesh-bbox-derived sole with points disabled beats the
     support-footprint sole on passive horizon without recreating early high
     force.
3. **The active 109-step failure is controller/standing dynamics after early
   contact force is fixed.**
   - Prediction: candidates can pass the 40-step `all` force gate but still
     fail active 140 around step 109.

## Stop Rules

- Stop before H200 if focused local tests fail.
- Stop a candidate if 40-step `all` peak ankle-roll force exceeds 1000 while
  upright.
- Do not run PPO.
- Do not tune the same scalar sphere radius family.

## Log

- 2026-05-13 Created after subtask008 showed a footprint-derived sole fixes
  early `all` force but fails passive horizon.
- 2026-05-13 H200 read-only mesh audit found the real ankle-roll visual mesh
  bbox:
  - left min/max `[-0.06584, -0.03774, -0.03541]` /
    `[0.14237, 0.03784, 0.02358]`;
  - right min/max `[-0.06584, -0.03784, -0.03541]` /
    `[0.14237, 0.03774, 0.02358]`;
  - source support spheres sit at x `-0.05` and `0.12`, y about `+/-0.03`,
    z `-0.03` with radius `0.005`.
- 2026-05-13 Added three patcher variants:
  - `ankle_roll_center_sole_keep_points`: explicit center box at
    `pos="0 0 -0.006"`, `size="0.035 0.020 0.006"`, with source point
    supports kept.
  - `ankle_roll_center_sole_no_points`: same center box, source point supports
    disabled with `contype=0 conaffinity=0`.
  - `ankle_roll_mesh_bbox_sole_no_points`: mesh-bbox horizontal box, source
    point supports disabled.
- 2026-05-13 Local focused tests passed:
  `22 passed in 0.88s`.
- 2026-05-13 H200 focused tests passed:
  `22 passed in 0.97s`.
- 2026-05-13 Generated subtask009 assets under
  `outputs/task023/sole_layout_assets/subtask009_layouts/assets/` with
  `missing=[]`, `errors=[]`, and `source_unchanged=true`.
- 2026-05-13 H200 Genesis 40-step `attitude + all` early-force gate:

  | Asset | First tilt/reset | Peak ankle-roll force | Result |
  | --- | ---: | ---: | --- |
  | `center_sole_keep_points` | none | 173.2 @ step 11 | pass |
  | `center_sole_no_points` | none | 198.0 @ step 2 | pass |
  | `mesh_bbox_sole_no_points` | none | 1188.2 @ step 2 | stop; bad broad sole contact |

- 2026-05-13 H200 Genesis 140-step validation for the two candidates that
  passed the early-force gate:

  | Probe | First tilt/reset | Peak ankle-roll force | Result |
  | --- | ---: | ---: | --- |
  | `center_sole_keep_points`, passive | 113 | 662.8 @ step 72 | preserves box-support passive horizon |
  | `center_sole_no_points`, passive | 87 | 198.0 @ step 2 | loses source/box-support horizon |
  | `center_sole_keep_points`, `attitude + all` | 109 | 383.4 @ step 112 | active collapse still unchanged |

## Review

Status: diagnostic_not_passed; point_support_semantics_identified.

- Hypothesis 1 passed. The difference between `center_sole_keep_points` and
  `center_sole_no_points` isolates the missing passive support: the old
  heel/toe point supports, not the center continuous box alone, provide the
  useful passive 113-step horizon.
- Hypothesis 2 failed in the broad-box form. The mesh-bbox horizontal sole with
  points disabled creates an early `all` force spike above 1000 at step 2.
  Bigger/visual-footprint box contact is not automatically more Genesis-ready.
- Hypothesis 3 passed. Once early force is controlled, active standing still
  collapses at 109. That is the same fixed-controller/standing-dynamics
  blocker seen in subtask007/subtask008.
- Root-cause refinement: Genesis needs both a sane continuous midfoot contact
  and heel/toe edge support semantics. A single continuous sole box, even when
  mesh-sized, cannot replace the old four support points under the current
  reset pose/controller.
- Next highest-value subtask: build a hybrid reviewed collision layout with
  continuous center/midfoot contact plus explicit heel/toe edge boxes, replacing
  point spheres without making one broad mesh-bbox slab. Gate remains:
  40-step `all` peak below 300, passive horizon at least 113, active horizon
  above 109 before any PPO retry.
