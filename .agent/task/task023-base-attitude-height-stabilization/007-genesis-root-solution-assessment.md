# Subtask 007: Genesis Root Solution Assessment

## Route

- Continue inside task023; no new top-level task.
- Genesis-only: no MuJoCo, no PPO, no downloads, no `/mnt/workspace*`
  writes/deletes, and no `GenesisG1SceneBackend` changes.
- Determine whether there is a root repair path rather than another scalar
  tuning path.

## Feedback Loop

```text
static XML hierarchy audit -> test representative support-geometry classes ->
compare early ankle-pitch force, real-controller force, and passive horizon
```

## Ranked Hypotheses

1. **Root fix is realistic foot support geometry, not sphere-size tuning.**
   - Prediction: a mesh-collision or box-support geometry changes the
     source/larger-spheres tradeoff: low early ankle-pitch force plus useful
     passive support.
2. **Box support is still too artificial/high-force.**
   - Prediction: it improves passive horizon but remains high-force under the
     active ankle-pitch/all-controller probes.
3. **Existing ankle-roll body assignment is physically expected.**
   - Prediction: XML hierarchy shows ankle-roll foot body is downstream of
     ankle pitch, so ankle-pitch-induced force on ankle-roll link is not an
     action mapping bug.
4. **No local XML patch is enough.**
   - Prediction: all representative support geometry classes either lose
     passive support or generate high early force; the root solution becomes
     importing/building a proper Genesis-ready collision asset.

## Stop Rules

- Stop before H200 if local patcher/probe tests fail.
- Stop a candidate if it produces >1000 early ankle-roll force while upright.
- Do not run PPO.
- Do not continue tuning the same sphere radius family.

## Log

- 2026-05-13 Created after subtask006 ruled out scalar sphere-radius tuning.
- 2026-05-13 Static source XML hierarchy audit, read on H200 without copying
  assets locally:
  - `left_ankle_pitch_link` is a parent body with
    `left_ankle_pitch_joint`;
  - `left_ankle_roll_link` is its child body with `left_ankle_roll_joint`;
  - the four source support spheres live under `left_ankle_roll_link`;
  - the right side mirrors the same structure.
  Therefore ankle-pitch motion loading ankle-roll-link contact is expected
  kinematics, not an action mapping bug. The issue is the foot/support contact
  geometry carried by that distal body.
- 2026-05-13 Implemented an extra patcher variant,
  `ankle_roll_mesh_collision`, which preserves the original visual mesh and
  appends a separate explicit colliding mesh geom under each ankle-roll link.
  Local focused tests passed:
  `PYTHONPATH=src python -m pytest
  tests/test_g1_ankle_roll_contact_patch.py
  tests/test_g1_base_attitude_height_stabilization.py -q -p no:cacheprovider`
  -> 22 passed. H200 focused tests after sync also passed -> 22 passed.
- 2026-05-13 Generated representative root-solution assets under
  `/root/agent_workspace/project/h200-locomotion-lab-task023-base-attitude-height-stabilization/outputs/task023/root_solution_assets/representative_support/`
  with `missing=[]`, `errors=[]`, and `source_unchanged=true`:
  - `assets/g1_27dof_nohand.ankle_roll_box_support.xml`;
  - `assets/g1_27dof_nohand.ankle_roll_mesh_collision.xml`.
- 2026-05-13 H200 Genesis-only early-force probes, physical GPU 1, logical
  `cuda:0`, `n_envs=64`, `steps=40`, `seed=120`, `mode=attitude`,
  `pose_profile=current`, `roll_allocation=hip_only_mirrored`,
  `roll_sign=normal`:

| Asset | Action joint group | Peak ankle-roll force | First >300 / >600 / >1000 | Clip ratio | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `ankle_roll_box_support` | `ankle_pitch` | 172.9 @ 12 | none / none / none | 0.125 | low-force; matches source-scale force |
| `ankle_roll_box_support` | `all` | 173.2 @ 11 | none / none / none | 0.625 | low-force under real controller target set |
| `ankle_roll_mesh_collision` | `ankle_pitch` | 243.7 @ 4 | none / none / none | 0.025 | low ankle-pitch-only force |
| `ankle_roll_mesh_collision` | `all` | 675.4 @ 4 | 4 / 4 / none | 0.650 | still high under coupled controller path |

- 2026-05-13 Because box support was the only representative that stayed
  low-force under `all`, ran 140-step validation:

| Probe | First tilt/reset | Peak ankle-roll force | First >300 / >600 / >1000 | Clip ratio | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `ankle_roll_box_support`, `mode=none` | 113/113 | 662.8 @ 72 | 53 / 64 / none | 0.000 | preserves passive support delay |
| `ankle_roll_box_support`, `mode=attitude`, `action_joint_group=all` | 109/109 | 383.4 @ 112 | 112 / none / none | 0.893 | early force fixed, but fixed controller still collapses |

## Review

Status: diagnostic_not_passed; root_direction_identified.

Root-solution assessment:

1. The structural issue is not that ankle-pitch action reports force on
   ankle-roll links. The XML hierarchy makes that expected: the foot/support
   body is downstream of the ankle-pitch joint.
2. The failed local patches were scalar point-contact tuning. Source 0.005
   point spheres are low-force but weak; 0.012 spheres help passive standing
   but create early active impulses; 0.006 loses the passive delay.
3. The representative continuous support surface (`ankle_roll_box_support`)
   is the first patch class that removes the early active impulse and preserves
   passive delay. It is a root-direction signal, not a final training asset.
4. The remaining 109-step active collapse on box support shows the next blocker
   is no longer the early contact impulse; it is fixed-controller/standing
   dynamics after the asset contact path is made coherent.

Therefore the practical root solution is two-stage:

- Build a proper Genesis-ready foot sole collision asset: continuous,
  physically located support under the distal ankle/foot body, explicit contact
  attrs, and reviewed dimensions. Do not continue tuning the same four
  point-sphere radii.
- Only after that asset passes the non-PPO gates, revisit standing controller
  or PPO. The current box support is a diagnostic proof of direction, not a
  production asset.

Ready-to-continue gate for the next subtask:

- A candidate foot contact asset should reproduce the box-support early-force
  behavior (`all` controller peak below 300 in the first 40 steps), preserve
  passive delay over source, and be justified by geometry/layout rather than an
  arbitrary support block.
