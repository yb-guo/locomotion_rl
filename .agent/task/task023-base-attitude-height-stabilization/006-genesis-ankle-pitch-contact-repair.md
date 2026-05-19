# Subtask 006: Genesis Ankle-Pitch Contact Repair

## Route

- Continue inside task023; do not create a new top-level task.
- Genesis-only: no MuJoCo route, no PPO, no walking, no downloads, no
  `/mnt/workspace*` writes/deletes, and no `GenesisG1SceneBackend` changes.
- Use the existing project-local XML patch generator. Start with a
  `larger_sphere_size` sweep because subtask005 isolated the failure to
  ankle-pitch targets interacting with `ankle_roll_larger_spheres` contact
  geometry.

## Feedback Loop

```text
generate task023-local sphere-size variants -> run 40-step ankle_pitch-only
Genesis probes -> compare early ankle-roll force against source and 0.012
larger_spheres -> run passive standing only for candidates that stay below
the early-force stop threshold
```

## Ranked Hypotheses

1. **The 0.012 sphere radius crosses a Genesis contact-force threshold.**
   - Prediction: 0.006 or 0.008 spheres keep `ankle_pitch` peak force below
     300 while 0.010/0.012 do not.
2. **There is a useful middle radius.**
   - Prediction: one smaller sphere radius reduces the early ankle-pitch
     impulse while still delaying passive collapse beyond the source step 88.
3. **The support-link semantics are wrong independent of radius.**
   - Prediction: all larger-than-source sphere radii either spike under
     `ankle_pitch` or lose the passive delay, so the next route must move or
     redesign contact support rather than tune sphere size.

## H200 Matrix

Generate variants under:

```text
outputs/task023/ankle_pitch_contact_repair/
```

Probe command family:

```text
mode=attitude
action_joint_group=ankle_pitch
roll_allocation=hip_only_mirrored
roll_sign=normal
steps=40
n_envs=64
seed=120
pose_profile=current
```

Initial variants:

| Variant label | Sphere size | Purpose |
| --- | ---: | --- |
| source | 0.005 source support | known low-force reference |
| larger_spheres_006 | 0.006 | near-source contact-size check |
| larger_spheres_008 | 0.008 | middle candidate |
| larger_spheres_010 | 0.010 | near-failing threshold check |
| larger_spheres_012 | 0.012 | existing failing reference |

Only run 120-step passive/attitude horizon probes for variants that keep
40-step `ankle_pitch` peak ankle-roll force below 300.

## Stop Rules

- Stop before H200 if patch generation reports missing/errors.
- Stop before runtime probes if generated assets are not under
  `/root/agent_workspace/project/.../outputs/task023`.
- Stop expanding after the first radius that gives both low early force and
  a passive delay over source.
- If no radius below 0.012 preserves passive support and avoids the early
  impulse, record sphere-size tuning as insufficient and move next to contact
  support redesign.

## Log

- 2026-05-13 Created per user direction to continue after subtask005.
- 2026-05-13 Local focused verification passed before H200 repair probes:
  `PYTHONPATH=src python -m pytest
  tests/test_g1_ankle_roll_contact_patch.py
  tests/test_g1_base_attitude_height_stabilization.py -q -p no:cacheprovider`
  -> 22 passed.
- 2026-05-13 Generated three task023-local repair variants under
  `/root/agent_workspace/project/h200-locomotion-lab-task023-base-attitude-height-stabilization/outputs/task023/ankle_pitch_contact_repair/`
  using the existing patch generator:
  - `sphere_size_006/assets/g1_27dof_nohand.ankle_roll_larger_spheres.xml`;
  - `sphere_size_008/assets/g1_27dof_nohand.ankle_roll_larger_spheres.xml`;
  - `sphere_size_010/assets/g1_27dof_nohand.ankle_roll_larger_spheres.xml`.
  All three reports had `status=completed`, `source_unchanged=true`,
  `missing=[]`, `errors=[]`, and generated XMLs rewrote the source relative
  `compiler.meshdir` to the absolute source mesh directory.
- 2026-05-13 H200 Genesis-only sphere-size matrix, physical GPU 1, logical
  `cuda:0`, `n_envs=64`, `steps=40`, `seed=120`, `mode=attitude`,
  `pose_profile=current`, `roll_allocation=hip_only_mirrored`,
  `roll_sign=normal`, `action_joint_group=ankle_pitch`:

| Asset label | Sphere size | Peak ankle-roll force | First >300 / >600 / >1000 | Clip ratio | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| source | 0.005 source | 172.9 @ 12 | none / none / none | 0.125 | low-force reference |
| `larger_spheres_006` | 0.006 | 304.0 @ 4 | 4 / none / none | 0.075 | near-threshold; no >600 |
| `larger_spheres_008` | 0.008 | 1291.4 @ 5 | 5 / 5 / 5 | 0.000 | fails early-force gate |
| `larger_spheres_010` | 0.010 | 1514.5 @ 6 | 6 / 6 / 6 | 0.000 | fails early-force gate |
| `ankle_roll_larger_spheres` | 0.012 | 1768.2 @ 7 | 7 / 7 / 7 | 0.000 | known failing reference |

- 2026-05-13 Because 0.006 was close to the low-force threshold, ran two
  validation probes:

| Probe | First tilt/reset | Peak ankle-roll force | First >300 / >600 / >1000 | Clip ratio | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `larger_spheres_006`, `mode=attitude`, `action_joint_group=all`, 40 steps | none/none | 866.2 @ 4 | 4 / 4 / none | 0.625 | real controller path still high-force |
| `larger_spheres_006`, `mode=none`, 140 steps | 89/89 | 558.0 @ 4 | 4 / none / none | 0.000 | loses the passive delay; essentially source-like horizon |

## Review

Status: diagnostic_not_passed; evidence_current.

Decision: sphere-size tuning is insufficient. There is a sharp force threshold
between 0.006 and 0.008, but the only near-low-force radius does not preserve
the useful passive delay and still produces high early force in the real
`action_joint_group=all` controller path. The 0.006 passive horizon is 89,
versus source 88 and the previous 0.012 larger-spheres passive horizon 106.

Root cause remains asset/contact support semantics rather than a scalar sphere
radius. The next Genesis-only repair route should change the support geometry
layout or support body assignment, not continue tuning the same ankle-roll-link
sphere radius. PPO standing remains closed.
