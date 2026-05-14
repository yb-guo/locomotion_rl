# Subtask 005: Genesis Contact Controller ABI Probe

## Route

- Continue inside task023; do not create a new top-level task.
- Genesis-only: no MuJoCo route in this subtask.
- Diagnosis-only: no PPO, no walking, no downloads, no `/mnt/workspace*`
  writes/deletes, and no `GenesisG1SceneBackend` changes.
- Keep the feedback loop small: identify which controller action group triggers
  the early `ankle_roll_larger_spheres` contact impulse seen in subtask004.

## Feedback Loop

Local loop:

```text
probe CLI exposes action-joint-group filtering -> fake Genesis tests verify
that only the requested group receives nonzero action targets -> summary keeps
contact link forces and reset pose audit
```

H200 loop:

```text
run 40-step larger_spheres attitude probes with one action group enabled at a
time -> compare first >300/>600/>1000 ankle-roll force, peak force step, link
force split, and reset/default qpos audit
```

## Ranked Hypotheses

1. **Leg pitch targets create the early support impulse.**
   - Prediction: `hip_pitch`, `ankle_pitch`, or combined `leg_pitch` produces
     the step-7 high force even when hip-roll targets are disabled.
2. **Hip-roll targets create the early support impulse.**
   - Prediction: `hip_roll` alone reproduces the high force, while leg-pitch
     groups do not.
3. **The impulse needs coupled roll+pitch targets.**
   - Prediction: single groups stay low, while `legs_no_ankle_roll` or `all`
     reproduces the high force.
4. **The impulse is reset/default-pose driven, not action-group driven.**
   - Prediction: all active groups behave similarly, and reset pose audit shows
     actual qpos diverging from the effective default before meaningful action
     differences.

## Planned H200 Matrix

Use `n_envs=64`, `steps=40`, `seed=120`, `mode=attitude`,
`pose_profile=current`, `roll_allocation=hip_only_mirrored`, `roll_sign=normal`,
physical GPU 1, logical `cuda:0`, and the task023
`ankle_roll_larger_spheres` asset.

| Action joint group | Purpose |
| --- | --- |
| `all` | short-window reference against subtask004 |
| `hip_roll` | isolate roll target path without ankle-roll actuation |
| `hip_pitch` | isolate proximal pitch target path |
| `ankle_pitch` | isolate distal pitch target path |
| `leg_pitch` | combined hip/knee/ankle pitch target path |
| `legs_no_ankle_roll` | coupled roll+pitch without ankle-roll targets |

Only run a source comparison if the larger-spheres branch identifies a single
action group that clearly reproduces the early impulse.

## Stop Rules

- Stop before H200 if local focused tests fail.
- Stop if a probe needs MuJoCo, PPO, downloads, or backend changes.
- Stop expanding the matrix once one action group reproduces >1000 ankle-roll
  force while other single groups stay below 300.
- If no group reproduces the impulse except coupled groups, record this as an
  action-coupling/contact-solver issue and do not continue tuning gains here.
- If every group reproduces the impulse, shift next work to reset/default-pose
  and asset contact semantics rather than controller mapping.

## Log

- 2026-05-13 Created per user direction to continue with the next subtask and
  exclude MuJoCo.
- 2026-05-13 Implemented scoped instrumentation in
  `g1_base_attitude_height_stabilization.py`:
  - `--action-joint-group` filter with `all`, `hip_roll`, `ankle_roll`,
    `hip_pitch`, `ankle_pitch`, `knee`, `leg_pitch`, and
    `legs_no_ankle_roll`;
  - peak/focus rows now include per-link forces for the existing ankle
    roll/pitch contact groups;
  - Genesis summaries include `reset_pose_audit` comparing actual reset qpos,
    effective default qpos, and profile default qpos.
- 2026-05-13 Local focused tests passed:
  `PYTHONPATH=src python -m pytest
  tests/test_g1_base_attitude_height_stabilization.py -q -p no:cacheprovider`
  -> 17 passed.
- 2026-05-13 Local related tests passed:
  `PYTHONPATH=src python -m pytest
  tests/test_g1_base_attitude_height_stabilization.py
  tests/test_vectorized_genesis_backend.py
  tests/test_g1_zero_action_standing_causality.py -q -p no:cacheprovider`
  -> 52 passed, 4 skipped.
- 2026-05-13 H200 focused tests passed after syncing the scoped patch:
  `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m pytest
  tests/test_g1_base_attitude_height_stabilization.py -q -p no:cacheprovider`
  -> 17 passed.
- 2026-05-13 H200 Genesis-only action-group matrix, physical GPU 1, logical
  `cuda:0`, `n_envs=64`, `steps=40`, `seed=120`, `mode=attitude`,
  `pose_profile=current`, `roll_allocation=hip_only_mirrored`,
  `roll_sign=normal`, `asset=ankle_roll_larger_spheres`:

| Action joint group | Peak ankle-roll force | First >300 / >600 / >1000 | Clip ratio | Peak link split | Result |
| --- | ---: | ---: | ---: | --- | --- |
| `all` | 1235.5 @ 7 | 7 / 7 / 7 | 0.800 | L 1235.5 / R 1230.5 | reproduces subtask004 early spike |
| `hip_roll` | 240.9 @ 7 | none / none / none | 0.100 | L 240.9 / R 240.7 | roll target alone is not sufficient |
| `hip_pitch` | 996.7 @ 7 | 7 / 7 / none | 0.775 | L 996.7 / R 992.8 | near-spike, but below >1000 |
| `ankle_pitch` | 1768.2 @ 7 | 7 / 7 / 7 | 0.000 | L 1768.2 / R 1763.0 | minimal single-group reproducer |
| `leg_pitch` | 1235.6 @ 7 | 7 / 7 / 7 | 0.800 | L 1235.6 / R 1230.7 | pitch path reproduces spike |
| `legs_no_ankle_roll` | 1235.6 @ 7 | 7 / 7 / 7 | 0.800 | L 1235.6 / R 1230.5 | no direct ankle-roll target needed |

  Peak-row state confirms the event is early and upright. For `ankle_pitch`,
  peak row is step 7 with upright 0.9999, tilt 0.0119, root height 0.7957,
  normalized action 0.1267, and `clipped=false`.
- 2026-05-13 Source comparison for the identified trigger:

| Asset | Action joint group | Peak ankle-roll force | First >300 / >600 / >1000 | Peak row |
| --- | --- | ---: | ---: | --- |
| source | `ankle_pitch` | 172.9 @ 12 | none / none / none | upright 1.0000, tilt 0.0081, normalized action 0.0166 |
| `ankle_roll_larger_spheres` | `ankle_pitch` | 1768.2 @ 7 | 7 / 7 / 7 | upright 0.9999, tilt 0.0119, normalized action 0.1267 |

  Reset pose audit is not the direct explanation: all larger-spheres group
  runs report actual-vs-effective-default max absolute error
  `2.38e-08`. The expected effective-vs-profile default deltas are the
  task018/task023 current tall-crouch pose differences, led by both knees
  (`0.549`) and ankle pitch (`0.293`).

## Review

Status: diagnostic_not_passed; evidence_current.

Decision: the minimal reproducer for the larger-spheres early force spike is
`ankle_pitch` action, not ankle-roll action, not hip-roll action, and not
clipping. The same `ankle_pitch`-only probe is low-force on the source asset,
so the root cause is a Genesis asset/contact geometry semantics mismatch:
larger ankle-roll contact geometry turns a modest ankle-pitch target into a
large ankle-roll-link support impulse while the robot is still upright.

Highest-value next route, still Genesis-only, is asset/contact repair rather
than controller mapping: audit or regenerate the foot/ankle contact geometry so
ankle-pitch targets do not load the ankle-roll links with >1000 force in the
first 10 steps. PPO standing is still not ready.
