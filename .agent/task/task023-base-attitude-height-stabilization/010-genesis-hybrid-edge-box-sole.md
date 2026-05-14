# Subtask 010: Genesis Hybrid Edge-Box Sole

## Route

- Continue inside task023; no new top-level task.
- Genesis-only. No MuJoCo, PPO, downloads, `/mnt/workspace*`
  writes/deletes, or `GenesisG1SceneBackend` changes.
- Replace point support spheres with explicit heel/toe edge boxes, then test
  whether support semantics survive without the old point contacts.

## Feedback Loop

```text
generate edge-box sole variants -> 40-step all-action force gate ->
140-step passive horizon -> 140-step active horizon
```

## Ranked Hypotheses

1. **Heel/toe edge boxes can replace point spheres.**
   - Prediction: disabling source point supports but adding heel/toe edge
     boxes keeps passive horizon near 113 while preserving low early force.
2. **Center + heel/toe hybrid is better than edge-only.**
   - Prediction: hybrid center pad plus edge boxes has equal or better passive
     horizon than edge boxes alone without recreating mesh-bbox early force.
3. **If active still resets around 109, the remaining blocker is controller
   dynamics, not early contact geometry.**
   - Prediction: a candidate can pass early force and passive horizon, but
     `attitude + all` still fails around 109.

## Stop Rules

- Stop before H200 if local focused tests fail.
- Stop a candidate if 40-step `all` peak ankle-roll force exceeds 1000 while
  upright.
- Do not run PPO.
- Do not run MuJoCo.

## Log

- 2026-05-13 Created after subtask009 showed old point supports carry
  heel/toe edge semantics: disabling them drops passive horizon from 113 to 87.
- 2026-05-13 Added two patcher variants:
  - `ankle_roll_edge_boxes_no_points`: disables source point supports and adds
    heel/toe edge boxes at x `-0.05` and `0.12`, z `-0.031`, size
    `0.010 0.030 0.004`.
  - `ankle_roll_hybrid_edge_boxes_no_points`: same edge boxes plus the center
    pad used by the box-support family.
- 2026-05-13 Local focused tests passed:
  `22 passed in 0.90s`.
- 2026-05-13 H200 focused tests passed:
  `22 passed in 0.30s`.
- 2026-05-13 Generated subtask010 assets under
  `outputs/task023/hybrid_edge_assets/subtask010_edge_boxes/assets/` with
  `missing=[]`, `errors=[]`, and `source_unchanged=true`.
- 2026-05-13 H200 Genesis 40-step `attitude + all` early-force gate:

  | Asset | First tilt/reset | Peak ankle-roll force | Result |
  | --- | ---: | ---: | --- |
  | `edge_boxes_no_points` | none | 173.6 @ step 11 | pass |
  | `hybrid_edge_boxes_no_points` | none | 173.6 @ step 12 | pass |

- 2026-05-13 H200 Genesis 140-step validation:

  | Probe | First tilt/reset | Peak ankle-roll force | Result |
  | --- | ---: | ---: | --- |
  | `edge_boxes_no_points`, passive | 89 | 173.7 @ step 13 | edge boxes alone do not replace full support |
  | `hybrid_edge_boxes_no_points`, passive | 116 | 743.7 @ step 77 | best passive support; beats 113 |
  | `hybrid_edge_boxes_no_points`, `attitude + all` | 108 | 173.6 @ step 11 | low force, but active horizon still fails |

## Review

Status: diagnostic_partial_not_passed.

- Hypothesis 1 partially passed. Heel/toe edge boxes alone are not enough
  (`89`), but they can replace point-sphere support only when combined with the
  center pad (`116`).
- Hypothesis 2 passed. Center + edge hybrid is the best Genesis contact asset
  candidate so far: it keeps early `all` force below 300 and improves passive
  horizon beyond box support/source-derived candidates.
- Hypothesis 3 passed. Active `attitude + all` still resets at `108` despite
  sane early force. The remaining blocker is not early contact impulse; it is
  controller/standing dynamics on a now-better contact asset.
- Decision: `ankle_roll_hybrid_edge_boxes_no_points` is the first candidate
  that is Genesis asset-ready enough for controller diagnosis, but not
  PPO-ready. Next highest-value subtask is fixed-controller retuning/ablation
  on this hybrid asset, especially clipping/max-delta and attitude gain, with
  passive `116` as the contact baseline.
