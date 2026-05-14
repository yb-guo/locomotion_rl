# Subtask 004: Controller Allocation Contact Coupling

## Route

- Continue inside task023; do not open a new task.
- Keep this diagnosis-only: no PPO, no walking, no downloads, and no
  `GenesisG1SceneBackend` changes.
- Extend the fixed-controller probe just enough to vary roll correction
  allocation and sign while keeping the same source and
  `ankle_roll_larger_spheres` assets from subtask002.
- Focus the H200 matrix on the key branch:
  why `attitude` delays source but regresses on `ankle_roll_larger_spheres`
  with high ankle-roll force.

## Feedback Loop

The local loop is:

```text
probe CLI accepts roll_allocation/roll_sign -> local tests verify generated
joint deltas, guarded command construction, summary metadata, and fake Genesis
actions
```

The H200 loop is:

```text
run bounded attitude-only variants against source and larger_spheres -> compare
first tilt/reset, clipping, ankle-roll force, ankle-pitch force, and top joint
errors against subtask002 baselines
```

## Ranked Hypotheses

1. **Wrong or over-broad roll allocation is driving ankle-roll contact spikes**
   - Prediction: removing ankle-roll from the attitude controller or using a
     mirrored hip-roll-only allocation reduces `larger_spheres` contact force
     without making first tilt earlier than the no-stabilizer baseline.
2. **Left/right roll sign semantics are mismatched**
   - Prediction: mirrored left/right roll allocation outperforms the current
     same-sign allocation on both assets, or at least removes the
     `larger_spheres` force spike.
3. **Ankle-roll contact remains dominant independent of mapping**
   - Prediction: all bounded roll allocations either fail near steps 88-109 or
     still create high ankle-roll force, which keeps the root cause at the
     asset/contact-controller semantics boundary.
4. **The source improvement was a clipped-delay artifact**
   - Prediction: variants that reduce clipping also remove the source-only
     first-tilt delay, showing the original `attitude` result was not a stable
     controller baseline.

## H200 Matrix

Use `mode=attitude`, `pose_profile=current`, `n_envs=64`, physical GPU 1,
logical `cuda:0`, and project-local generated `ankle_roll_larger_spheres`.

Start with:

| Asset | Roll allocation | Roll sign | Purpose |
| --- | --- | --- | --- |
| source | `hip_ankle_same` | `normal` | existing subtask002 reference |
| source | `hip_only_mirrored` | `normal` | remove ankle-roll actuation |
| source | `ankle_only_mirrored` | `normal` | isolate ankle-roll actuation |
| source | `hip_ankle_mirrored` | `normal` | test left/right sign semantics |
| larger_spheres | `hip_ankle_same` | `normal` | existing high-force reference |
| larger_spheres | `hip_only_mirrored` | `normal` | test if force spike is ankle actuation |
| larger_spheres | `ankle_only_mirrored` | `normal` | reproduce/isolate spike |
| larger_spheres | `hip_ankle_mirrored` | `normal` | test sign semantics with cleaner contact |

Only add `roll_sign=inverted` if mirrored normal produces a clear improvement
or a clear directional contradiction.

## Stop Rules

- Stop before H200 if local focused tests fail.
- Stop a candidate family if ankle-roll max force exceeds task023
  `larger_spheres+attitude` (`~1236`) without a first-tilt improvement.
- If hip-only mirrored improves both assets with sane contact force, do not
  expand the matrix; record controller allocation as the next implementation
  direction.
- If all allocations fail near current horizons, keep PPO closed and record
  that asset/contact-controller semantics, not a simple fixed stabilizer, still
  dominate.
- Do not mark task023 passed unless a candidate is reproduced and clearly
  exceeds source and larger-spheres baselines with sane contact force.

## Log

- 2026-05-13 Created after task023 final diagnostic result, per user direction
  to keep the next diagnosis inside this task.
- 2026-05-13 Local implementation started by extending
  `g1_base_attitude_height_stabilization` with bounded roll allocation/sign
  variants while preserving the previous default behavior.
- 2026-05-13 Local focused verification passed:
  `PYTHONPATH=src python -m pytest
  tests/test_g1_base_attitude_height_stabilization.py -q -p no:cacheprovider`
  -> 16 passed.
- 2026-05-13 Local related verification passed:
  `PYTHONPATH=src python -m pytest
  tests/test_g1_base_attitude_height_stabilization.py
  tests/test_vectorized_genesis_backend.py
  tests/test_g1_zero_action_standing_causality.py -q -p no:cacheprovider`
  -> 51 passed, 4 skipped.
- 2026-05-13 H200 focused verification after syncing the scoped patch passed
  through `run_guarded.sh` with `CUDA_VISIBLE_DEVICES=1`:
  `PYTHONPATH=src python -m pytest
  tests/test_g1_base_attitude_height_stabilization.py -q -p no:cacheprovider`
  -> 16 passed.
- 2026-05-13 H200 partial source-only allocation evidence, physical GPU 1,
  logical `cuda:0`, `n_envs=64`, `mode=attitude`, `pose_profile=current`,
  `seed=120`, `steps=140`:

| Asset | Roll allocation | First tilt/reset | Clipping ratio | Ankle-roll max force | Result |
| --- | --- | ---: | ---: | ---: | --- |
| source | `hip_ankle_same` | 109 | 0.875 | 258.7 confirm / 652.3 first run | existing subtask002 reference |
| source | `hip_only_mirrored` | 109 | 0.893 | 907.8 | no horizon improvement; higher force than reference confirm |
| source | `ankle_only_mirrored` | 109 | 0.893 | 383.4 | no horizon improvement; lower than hip-only but not a fix |

  New source summaries:
  `outputs/task023/base_attitude_height_stabilization/source_attitude_hip_only_mirrored_n64_s140_pose_current/summary.json`
  and
  `outputs/task023/base_attitude_height_stabilization/source_attitude_ankle_only_mirrored_n64_s140_pose_current/summary.json`.
- 2026-05-13 Remaining H200 matrix rows are pending. A subsequent SSH
  escalation request for the next same-family H200 probe was rejected by the
  automatic approval reviewer before command execution, so the route stopped
  without running the larger-spheres allocation branch.
- 2026-05-13 User approved continuing the guarded H200 probes. Ran
  `source_attitude_hip_ankle_mirrored_n64_s140_pose_current`: source still
  failed at first tilt/reset 109, clipping ratio 0.893, ankle-roll max force
  567.4. This confirms source first-tilt horizon is insensitive to tested
  same-vs-mirrored roll allocation.
- 2026-05-13 Ran the critical larger-spheres branch
  `larger_spheres_attitude_hip_only_mirrored_n64_s140_pose_current`:
  first tilt/reset 105 versus no-stabilizer baseline 106, clipping ratio
  0.943, ankle-roll max force 1235.5, ankle-pitch max force 0.0. Because this
  high-force regression appears even with ankle-roll actuation removed from the
  controller, the spike is not explained by direct ankle-roll target deltas
  alone. It is a contact/controller coupling: active attitude feedback changes
  body state/load path enough that the larger ankle-roll contact patch receives
  a large force impulse. Stop rule triggered; do not expand to more
  larger-spheres allocation rows until a lower-force controller family is
  designed.
- 2026-05-13 Added force-event timeline and controller start/stop gating
  instrumentation. Local focused and related tests still passed, then H200
  focused tests passed after syncing the scoped patch.
- 2026-05-13 H200 force-event/gating matrix, physical GPU 1, logical
  `cuda:0`, `n_envs=64`, `pose_profile=current`, `seed=120`, `steps=140`:

| Run | First tilt/reset | Peak ankle-roll force | First >300 / >600 / >1000 | Mean ankle-roll force | Clip ratio | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `source_none_force_timeline_n64_s140_pose_current` | 88/88 | 173.0 @ 10 | none / none / none | 95.1 | 0.000 | source passive collapse; no early force spike |
| `source_attitude_hip_only_mirrored_force_timeline_n64_s140_pose_current` | 109/109 | 383.4 @ 112 | 112 / none / none | 126.4 | 0.893 | source delay is real; peak force happens after tilt/reset onset |
| `larger_spheres_none_force_timeline_n64_s140_pose_current` | 106/106 | 324.1 @ 111 | 111 / none / none | 120.1 | 0.000 | larger-spheres passive delay; force peak is collapse aftermath |
| `larger_spheres_attitude_hip_only_mirrored_n64_s140_pose_current` | 105/105 | 1235.5 | n/a | 139.0 | 0.943 | old summary; confirms high-force regression without ankle-roll actuation |
| `larger_spheres_attitude_hip_only_mirrored_delta004_n64_s140_pose_current` | none/none | 1769.6 @ 7 | 7 / 7 / 7 | 174.7 | 0.993 | early controller/contact impulse while upright; 140-step horizon survives |
| `larger_spheres_attitude_hip_only_mirrored_delta002_n64_s140_pose_current` | 116/116 | 1073.9 @ 7 | 7 / 7 / 7 | 133.3 | 0.950 | smaller action still triggers early impulse; modest horizon delay |
| `larger_spheres_attitude_hip_only_mirrored_start080_n64_s140_pose_current` | 105/105 | 241.0 @ 7 | none / none / none | 116.1 | 0.429 | no early spike when controller is disabled until step 80 |
| `larger_spheres_attitude_hip_only_mirrored_stop080_n64_s140_pose_current` | 106/106 | 1235.7 @ 7 | 7 / 7 / 7 | 131.1 | 0.514 | early spike appears with controller enabled, but stopping at 80 reverts to passive-like failure |

  Evidence lives under
  `/root/agent_workspace/project/h200-locomotion-lab-task023-base-attitude-height-stabilization/outputs/task023/base_attitude_height_stabilization/`.
- 2026-05-13 Event-row inspection:
  - `source + attitude hip_only_mirrored` peak force is at step 112, after
    first tilt/reset step 109, with upright already 0.071 and root height
    0.093. This is collapse aftermath, not the source improvement mechanism.
  - `larger_spheres + attitude hip_only_mirrored stop080` peak force is at
    step 7 while upright is 0.9995 and root height is 0.797. The same step in
    `start080` has controller disabled and only 241.0 ankle-roll force.
  - Therefore the larger-spheres force spike is caused by early active
    controller/PD/contact coupling, but stopping the controller before collapse
    makes the first tilt/reset match the passive 106-step path. The spike is a
    strong symptom of semantic mismatch, not by itself the sole collapse cause.

## Review

Status: diagnostic_not_passed; evidence_current.

Root-cause hypothesis tree:

1. **Primary: Genesis contact geometry and controller semantics are coupled.**
   The ankle-roll-named contact geometry is the support/contact path;
   ankle-pitch contact force stays 0.0. Larger spheres improve passive horizon
   from 88 to 106, but when an active attitude controller is enabled from the
   first few steps, the same contact patch receives >1000 force while the base
   is still upright. This is not explained by ankle-roll joint actuation because
   `hip_only_mirrored` reproduces it.
2. **Secondary: the fixed controller is clipped and too coarse for this ABI.**
   The controller remains clipped in 89-99% of active steps across tested
   source/larger-spheres variants. Action scale is not linearly causal:
   `max_joint_delta=0.04` survives 140 steps despite the largest early impulse,
   while `0.02` delays to 116 and `0.08` fails around 105-106. This points to
   solver/contact-state sensitivity, not a clean proportional gain issue.
3. **Unlikely: a simple left/right roll sign or ankle-roll action mapping bug.**
   Source first-tilt horizon stays 109 across same/mirrored and hip/ankle
   allocation variants. Larger-spheres still spikes with ankle-roll actuation
   removed from roll feedback.
4. **Unproven but still relevant: reset/default pose mismatch.**
   Task023 already fixed projected-gravity reset semantics and
   `pose_profile=current`. The remaining failures happen with those fixes, so
   reset semantics are not the direct cause. The first 10-step settling
   transient still deserves ABI-level inspection before PPO.
5. **Open simulator mismatch: MuJoCo/SONIC vs Genesis.**
   SONIC/MuJoCo can move because its policy, actuator scaling, contact solver,
   default qpos, and XML semantics are internally matched. Genesis PPO standing
   is failing before PPO because the imported asset/contact geometry and action
   target semantics are not yet a matched standing ABI.

Highest-value next task, superseded by the user constraint in subtask005:
build a small **Genesis-only contact-controller ABI probe** before retrying
PPO. The probe should separate asset geometry from controller semantics with
one-step and short-horizon sweeps, not training.

Concrete next experiments:

- Per-geom contact audit for source and larger-spheres: contact body, geom
  name, force, normal, and active step for the first 15 steps and around first
  tilt.
- One-joint-group target sweep in Genesis: hold reset qpos and command only
  hip roll, hip pitch, knee, ankle pitch, ankle roll, and upper-body defaults;
  record immediate contact impulse and root acceleration.
- Short action-scale/gain sweep with `controller_start_step=0/20/80` and
  `controller_stop_step=20/80`, using only 0-40 step event windows first.
- Reset/default qpos audit: compare profile default angles, `pose_profile`
  reset qpos, Genesis actual qpos after reset, and first-step PD target for
  every actuator.
- Genesis-only source-vs-larger-spheres static drop/PD-target replay, comparing
  contact activation order and actuator target convention without downloading
  anything.

Stop rules for the next route:

- Stop if a probe requires PPO, downloads, or `GenesisG1SceneBackend` changes.
- Stop a controller family if it produces >1000 ankle-roll force while upright
  without extending both source and larger-spheres horizons.
- Stop treating ankle-roll as the action bug unless a one-joint ankle-roll
  target sweep reproduces the force spike while hip-only/upper-body sweeps do
  not.
- Do not expand long-horizon sweeps until the first 0-15 step contact impulse
  source is identified.

Ready to retry PPO standing only when a non-PPO gate passes: source and the
chosen training asset survive a confirmed fixed-controller or zero-action
standing horizon well beyond the current 106-116 step band, with no early
upright >1000 ankle-roll force impulse, bounded clipping, documented action
mapping, and contact evidence showing both the support geometry and actuator
targets have coherent Genesis semantics.
