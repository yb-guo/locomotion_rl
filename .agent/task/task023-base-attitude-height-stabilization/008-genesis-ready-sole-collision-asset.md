# Subtask 008: Genesis-Ready Sole Collision Asset

## Route

- Continue task023; no new top-level task.
- Genesis-only. No MuJoCo, PPO, downloads, `/mnt/workspace*`
  writes/deletes, or `GenesisG1SceneBackend` changes.
- Build first proper sole-collision candidate, not another sphere-radius tune.

## Feedback Loop

```text
derive sole box from source support-sphere footprint -> disable point supports
-> generate task023-local asset -> run early-force and horizon probes
```

## Hypotheses

1. **Footprint-derived continuous sole fixes early impulse.**
   - Prediction: `action_joint_group=all`, 40-step peak ankle-roll force below
     300 while upright.
2. **Footprint-derived sole preserves support better than source.**
   - Prediction: passive horizon beats source step 88.
3. **If it fails, root fix still needs better sole layout.**
   - Prediction: high early force or weak passive horizon means derive-from
     support spheres is not enough; need reviewed foot geometry dimensions.

## Stop Rules

- Stop before H200 if local focused tests fail.
- Stop if generated patch reports missing/errors.
- Stop if early 40-step `all` peak exceeds 1000.
- Do not run PPO.

## Log

- 2026-05-13 Created per user request to build Genesis-ready continuous foot
  sole collision asset.
- 2026-05-13 Added `ankle_roll_sole_collision` patcher variant. It derives one
  low box sole per ankle-roll link from the source support-sphere footprint,
  disables the old point support spheres with `contype=0 conaffinity=0`, and
  preserves the visual mesh as non-colliding.
- 2026-05-13 Local focused tests passed:
  `22 passed in 0.55s`.
- 2026-05-13 H200 focused tests passed:
  `22 passed in 1.00s`.
- 2026-05-13 Generated task-local asset with `missing=[]`, `errors=[]`, and
  `source_unchanged=true`:
  `outputs/task023/sole_collision_assets/footprint_sole/assets/g1_27dof_nohand.ankle_roll_sole_collision.xml`.
  The real source footprint generated each sole box at
  `pos="0.035 0 -0.031"` and `size="0.09 0.035 0.004"`.
- 2026-05-13 H200 Genesis probes:

  | Probe | First tilt/reset | Peak ankle-roll force | Result |
  | --- | ---: | ---: | --- |
  | `ankle_pitch`, 40 steps | none | 389.0 @ step 3 | Better than larger-spheres 1768.2, but still above 300 |
  | `all`, 40 steps | none | 175.5 @ step 9 | Passes early all-action force gate |
  | passive, 140 steps | 87 | 435.1 @ step 3 | Fails support-horizon goal; not better than source 88 |
  | `attitude + all`, 140 steps | 109 | 175.5 @ step 9 | Low force, but horizon equals source attitude-only 109 |

## Review

Status: diagnostic_partial_not_passed.

- Hypothesis 1 partially passed. The continuous sole removes the severe
  all-action early impulse and keeps `all` below 300 for 40 steps, but the
  isolated `ankle_pitch` probe still reaches 389.0 at step 3.
- Hypothesis 2 failed. Passive standing does not beat source; it resets at
  step 87 versus the accepted source baseline around step 88.
- Hypothesis 3 is confirmed. Deriving a box only from the existing support
  sphere footprint is not enough. It fixes one bad contact-force mode but does
  not create a PPO-ready standing asset.
- Decision: keep this variant as a useful Genesis force-control evidence
  point, not as the final training asset. The next repair must use reviewed
  foot/sole geometry semantics, not only the old support-sphere footprint.
