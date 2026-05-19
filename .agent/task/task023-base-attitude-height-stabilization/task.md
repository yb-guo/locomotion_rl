# Task 023: Base Attitude Height Stabilization

## Goal

Determine whether an explicit base attitude/height stabilization controller can
arrest the G1 standing collapse that task020-task022 isolated away from PPO
plumbing.

This task is diagnosis-first. The primary pass/fail signal is zero-action or
fixed-controller standing stability under Genesis, not PPO reward.

## Scope

- Branch: `codex/task023-base-attitude-height-stabilization`.
- Worktree:
  `../_worktrees/h200-locomotion-lab-task023-base-attitude-height-stabilization`.
- Remote project:
  `/root/agent_workspace/project/h200-locomotion-lab-task023-base-attitude-height-stabilization`.
- Use prepared G1 source asset only:
  `/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_27dof_nohand.xml`.
- Regenerate any task022-style comparison asset under this task's project
  outputs. Do not depend on task022 remote output paths.
- Compare at least:
  - source asset;
  - clean `ankle_roll_larger_spheres` contact comparison asset;
  - no stabilizer baseline;
  - attitude-only, height-only, and attitude+height stabilizer candidates.

## Non-Goals

- No PPO.
- No walking or `vx_yaw`.
- No LocoFormer.
- No ONNX.
- No rendering/GIF/video.
- No downloads of assets, datasets, checkpoints, or upstream repos.
- No writes/deletes under `/mnt/workspace` or `/mnt/workspace1`.
- No change to `GenesisG1SceneBackend`.
- Do not promote `ankle_roll_box_support` as a training asset in this task.

## H200 Protocol

Remote commands must use:

```bash
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'
```

All remote code, outputs, generated XML, and intermediate files must stay under:

```text
/root/agent_workspace/project
```

Default GPU metadata:

```text
CUDA_VISIBLE_DEVICES=1
physical_gpu=1
logical_cuda_device=cuda:0
```

## Diagnosis Context

Task020 showed the standing PPO gate is blocked by env/contact/passive-standing
dynamics, not PPO plumbing.

Task021 localized the immediate contact path to `left/right_ankle_roll_link`.

Task022 showed contact geometry matters but is not sufficient:

- source first tilt: step 88;
- `ankle_roll_friction_attrs`: step 88, no delay;
- `ankle_roll_larger_spheres`: step 106 and confirm step 106, lower ankle-roll
  contact force than source;
- `ankle_roll_box_support`: step 113 and confirm step 113, but much higher
  ankle-roll contact force;
- all variants still failed from tilt/reset.

## Diagnose Loop

### Feedback Loop

Build a deterministic fixed-controller probe that can run locally without
Genesis for tests and on H200 with Genesis for evidence.

The tool should report:

- effective asset path;
- stabilizer mode and gains;
- root height/upright timeline;
- first tilt/reset step;
- top joint errors;
- ankle-roll and ankle-pitch contact trace when enabled;
- whether improvement is physical stability or only delayed reset.

Pass signal:

```text
A bounded fixed controller reproducibly extends standing horizon beyond both
source and larger-spheres baselines while keeping contact forces sane and not
hiding reset semantics.
```

Failure signal:

```text
All controller candidates still collapse near the task022 horizons, produce
unbounded contact forces, or only mask the reset condition.
```

### Ranked Hypotheses

1. **Base attitude feedback is the missing stabilizer**
   - Prediction: roll/pitch feedback through leg joints delays tilt more than a
     pure contact patch.
2. **Height regulation is needed in addition to attitude**
   - Prediction: attitude-only improves upright but still loses root height;
     attitude+height improves both.
3. **Contact patch and controller interact**
   - Prediction: the same controller performs better on
     `ankle_roll_larger_spheres` than on the source asset.
4. **Asset semantics remain the dominant blocker**
   - Prediction: controller candidates cannot avoid collapse without causing
     high ankle-roll forces or unnatural joint targets.

## Stop Rules

- First implement a local deterministic probe and tests; do not run H200 before
  the local command passes.
- Do not tune PPO or train a policy in this task.
- Run source and `ankle_roll_larger_spheres` baselines before controller
  candidates.
- Change one controller family at a time: attitude-only, height-only, then
  attitude+height.
- If a candidate improves first tilt by at least 20 policy steps, rerun it once
  before treating it as evidence.
- Stop if a controller requires changing inertials, source assets, or
  `GenesisG1SceneBackend`.
- Stop if contact force spikes exceed task022 box-support levels without a
  clear stability benefit.

## Route

1. `000-contract-and-feedback-loop.md`
2. `001-fixed-controller-probe.md`
3. `002-h200-controller-matrix.md`
4. `003-review-and-decision.md`
5. `004-controller-allocation-contact-coupling.md`
6. `005-genesis-contact-controller-abi-probe.md`
7. `006-genesis-ankle-pitch-contact-repair.md`
8. `007-genesis-root-solution-assessment.md`
9. `008-genesis-ready-sole-collision-asset.md`
10. `009-genesis-sole-layout-ablation.md`
11. `010-genesis-hybrid-edge-box-sole.md`
12. `011-hybrid-asset-controller-ablation.md`
13. `012-sonic-runability-check.md`
14. `013-ppo-hybrid-asset-switch.md`
15. `014-hybrid-ppo-standing-causality.md`
16. `015-base-frame-semantics-probe.md`
17. `016-zero-action-support-trace.md`

## Acceptance

- Router creates task/subtask docs before coding.
- Coding subagent implements scoped code changes.
- Read-only reviewer reviews code and evidence.
- Local focused tests pass.
- H200 guarded evidence compares source and `ankle_roll_larger_spheres`.
- Decision states one of:
  - fixed stabilizer is viable enough to become a standing-controller baseline;
  - stabilizer helps partially but asset semantics still dominate;
  - stabilizer is insufficient and next task should target upstream asset or
    model/controller semantics.

## Log

- 2026-05-13 Created after task022 showed contact patching helps partially but
  does not stabilize passive standing.
- 2026-05-13 Completed subtask001 local deterministic feedback loop and tests.
- 2026-05-13 Completed subtask002 H200 guarded matrix after fixing two runner
  equivalence issues: projected-gravity upright/reset semantics and
  `pose_profile=current` default reset pose. H200 evidence reproduced source
  baseline first tilt/reset step 88 and `ankle_roll_larger_spheres` baseline
  step 106.
- 2026-05-13 Matrix decision before final review: fixed attitude feedback gives
  a reproducible source-only delay to step 109, but height feedback is
  negligible, attitude+height is inconsistent/high-force, and the
  larger-spheres plus attitude combination regresses with very high ankle-roll
  force. The controller does not arrest collapse.
- 2026-05-13 Reopened task023 for a scoped continuation instead of creating a
  new task. Subtask004 targets the key branch: why the same attitude
  controller delays source but regresses on `ankle_roll_larger_spheres` with
  high ankle-roll force.
- 2026-05-13 Subtask004 local implementation and verification passed. H200
  focused tests passed after syncing to the remote task023 worktree. Two
  source allocation probes ran: `hip_only_mirrored` and
  `ankle_only_mirrored` both kept first tilt/reset at 109, so neither improved
  the source horizon beyond the original attitude result. The remaining
  larger-spheres allocation branch is still pending after SSH escalation for
  the next same-family H200 probe was rejected before execution.
- 2026-05-13 User approved continuing. Completed one more source allocation
  row and the critical larger-spheres row. `source + hip_ankle_mirrored` still
  failed at 109. `larger_spheres + hip_only_mirrored` failed at 105 with
  ankle-roll max force about 1235.5 despite no ankle-roll actuation in the roll
  controller. This supports contact/controller coupling rather than a simple
  ankle-roll action mapping bug. Stop rule triggered; larger-spheres allocation
  expansion is closed until a lower-force controller family exists.
- 2026-05-13 Added force-event timeline and controller start/stop gating to
  the fixed-controller probe. Local focused tests passed (`16 passed`), local
  related tests passed (`51 passed, 4 skipped`), and H200 focused tests passed
  (`16 passed`) after syncing the scoped patch.
- 2026-05-13 Completed source/larger-spheres force-event probes. Source
  `none` failed at 88 with no >300 ankle-roll force; source
  `attitude hip_only_mirrored` delayed to 109 and only exceeded 300 after the
  collapse path had already started. Larger-spheres `none` failed at 106 and
  only exceeded 300 after collapse. Larger-spheres `attitude hip_only_mirrored`
  with early controller enabled produced >1000 ankle-roll force at step 7 while
  still upright; disabling the controller until step 80 removed the early spike
  but did not improve the 105-step failure, while stopping the controller at
  step 80 kept the early spike but reverted to passive-like failure at 106.
  Conclusion: early high force is controller/contact coupling specific to the
  larger-spheres support geometry, but it is not the sole collapse cause.
- 2026-05-13 Opened subtask005 per user direction to continue with the next
  subtask and exclude MuJoCo. The new route is Genesis-only action-group and
  reset-pose ABI diagnosis.
- 2026-05-13 Subtask005 local focused tests passed (`17 passed`), local
  related tests passed (`52 passed, 4 skipped`), and H200 focused tests passed
  (`17 passed`) after syncing the scoped patch. H200 Genesis-only action-group
  matrix found the minimal early-force reproducer: on
  `ankle_roll_larger_spheres`, `ankle_pitch` action alone produced ankle-roll
  peak force 1768.2 at step 7 while upright 0.9999 and unclipped. `hip_roll`
  alone stayed low at 240.9. The same source-asset `ankle_pitch`-only probe
  stayed low at 172.9. Reset pose audit showed actual-vs-effective-default
  max error only `2.38e-08`. Conclusion: the larger-spheres asset/contact
  geometry turns modest ankle-pitch targets into large ankle-roll-link support
  impulses; this is not an ankle-roll action mapping bug and not clipping.
- 2026-05-13 Opened subtask006 per user direction to continue. Route is
  Genesis-only asset/contact repair. First repair matrix is a controlled
  sphere-size sweep around the known source `0.005` and failing larger-spheres
  `0.012` contact geometry.
- 2026-05-13 Subtask006 local focused tests passed (`22 passed`). Generated
  task023-local sphere-size repair variants `0.006`, `0.008`, and `0.010` with
  `missing=[]`, `errors=[]`, and `source_unchanged=true`. H200 Genesis-only
  `ankle_pitch` probes showed a sharp force threshold: source peak 172.9,
  `0.006` peak 304.0, `0.008` peak 1291.4, `0.010` peak 1514.5, and existing
  `0.012` peak 1768.2. The near-threshold `0.006` variant failed validation:
  real `action_joint_group=all` still had peak 866.2, and passive standing
  failed at 89 versus source 88 and 0.012 passive 106. Conclusion:
  sphere-radius tuning is insufficient; next Genesis-only repair must change
  support geometry layout or support body assignment.
- 2026-05-13 Opened subtask007 per user request to look for a root solution.
  Route is Genesis-only and compares representative support-geometry classes,
  not more sphere-radius tuning.
- 2026-05-13 Subtask007 static XML audit showed ankle-roll foot/support body
  is downstream of the ankle-pitch joint, so ankle-pitch-induced force on
  ankle-roll links is expected kinematics, not an action mapping bug. Added
  `ankle_roll_mesh_collision` as a representative mesh-collision patcher
  variant; local and H200 focused tests passed (`22 passed`). H200 root
  representative probes found `ankle_roll_box_support` removes the early active
  impulse under both `ankle_pitch` and `all` action groups (peak about 173,
  no >300), while preserving passive support delay at 113. However,
  `box_support + attitude + all` still fails at 109 with moderate post-collapse
  force. Conclusion: the root direction is a proper continuous Genesis foot
  sole collision asset, not sphere-radius tuning. Box support proves the
  direction but is not a final training asset; PPO remains closed.
- 2026-05-13 Opened subtask008 per user request to build a Genesis-ready
  continuous sole collision asset. First candidate derives a low sole box from
  the source support-sphere footprint and disables the old point supports.
- 2026-05-13 Subtask008 added `ankle_roll_sole_collision`, passed local and
  H200 focused tests (`22 passed` both), and generated a task-local sole-box
  asset with `missing=[]`, `errors=[]`, and `source_unchanged=true`. H200
  Genesis probes showed the candidate fixes the severe all-action early force
  mode (`all`, 40-step peak 175.5, no >300), but does not fix standing
  dynamics: passive reset remained at 87 and `attitude + all` reset remained
  at 109, matching the source attitude-only horizon.
- 2026-05-13 Opened subtask009 to separate point-support retention from
  continuous sole footprint/layout. H200 read-only mesh audit showed the
  ankle-roll visual mesh horizontal bbox is only moderately larger than the
  source support-sphere footprint, so the next candidates compare center pad
  with points kept, center pad with points disabled, and mesh-bbox sole with
  points disabled.
- 2026-05-13 Subtask009 added three sole-layout variants and passed local/H200
  focused tests (`22 passed` both). H200 Genesis gates found:
  `center_sole_keep_points` passes 40-step `all` force (173.2) and preserves
  passive horizon 113, `center_sole_no_points` passes early force but passive
  horizon falls to 87, and `mesh_bbox_sole_no_points` fails early force with
  1188.2 at step 2. Active `center_sole_keep_points` still resets at 109.
  Conclusion: point/edge support semantics are necessary; a broad continuous
  slab is bad, and a center pad alone is insufficient once point supports are
  removed.
- 2026-05-13 Opened subtask010 to replace point spheres with explicit
  heel/toe edge boxes and test edge-only versus center+edge hybrid support.
- 2026-05-13 Subtask010 added `edge_boxes_no_points` and
  `hybrid_edge_boxes_no_points`, passed local/H200 focused tests (`22 passed`
  both), and generated clean assets. Both passed 40-step `all` early-force
  gate around 173.6. Edge boxes alone failed passive horizon at 89, while the
  center+edge hybrid improved passive horizon to 116. Active
  `attitude + all` on the hybrid stayed low-force but reset at 108. Conclusion:
  hybrid contact asset is the best Genesis asset candidate so far, but active
  controller dynamics now dominate.
- 2026-05-13 Opened subtask011 to freeze
  `ankle_roll_hybrid_edge_boxes_no_points` and ablate fixed-controller
  `max_joint_delta` plus attitude gain on H200 Genesis.
- 2026-05-13 Subtask011 controller matrix found continuous attitude control is
  still the wrong family on the hybrid asset. Lowering `max_joint_delta` only
  improves reset to 113/114, below passive 116, and gain reduction alone does
  not help. Timing probes found the best result so far:
  default attitude controller stopped at step 40 reaches reset 122, confirmed
  on seeds 120 and 121, with no >1000 force. This suggests a short early pose
  settling pulse helps, while sustained clipped feedback destabilizes the
  later standing dynamics.
- 2026-05-13 Opened subtask012 after user asked for a SONIC runability check.
  Scope is existing official SONIC artifacts only: no download, no training,
  no PPO, and no upstream edits.
- 2026-05-13 Subtask012 ran official SONIC smoke on H200 using existing task002
  artifacts. Preflight found the MuJoCo XML, deploy script, ONNX/config files,
  and reference motions. Headless sim-only ran until timeout without XML/MuJoCo
  crash. Dual sim+deploy reached native policy-load, loaded 13 reference motion
  folders, and passed ONNX hash checking, but failed at TensorRT runtime
  creation with `CUDA initialization failure with error: 35` and segfault
  (`DEPLOY_EXIT_CODE=139`), unchanged with `CUDA_VISIBLE_DEVICES=1`. H200 shows
  driver `570.195.03`, CUDA `12.8`, toolkit `12.8.61`, GPU 1 visible, and both
  `libcudart.so.12` and `libcudart.so.13` visible to the loader. No residual
  SONIC processes remained. Note: `deploy.sh` ran its built-in CMake/build
  step in the existing upstream run directory; no source/download/checkpoint or
  `/mnt/workspace*` mutation was performed by this task.
- 2026-05-13 Followed up after user clarified the goal was asset health via the
  old successful SONIC route. Task002 shows the old full sim2sim pass depended
  on extracted TensorRT `10.13.3.9-1+cuda12.9` runtime libraries. That
  extraction is no longer present, and no cached CUDA12.9 TensorRT deb was
  found locally or in the checked H200 task/cache paths; current system
  TensorRT is `10.13.3.9-1+cuda13.0`, the documented error-35 path. The
  asset-side check still passed: direct MuJoCo load/step of
  `g1_29dof.xml` produced finite state (`nq=36`, `nv=35`, `nu=29`,
  `ngeom=76`), and default task002-style `xvfb-run` sim loop ran for 60s until
  timeout without XML/mesh/MuJoCo crash. Full policy/control rerun remains
  blocked by the missing CUDA12.9 TensorRT runtime, not by asset evidence.
- 2026-05-13 Opened subtask013 to switch PPO smoke asset selection to the best
  current Genesis candidate, `ankle_roll_hybrid_edge_boxes_no_points`, without
  running PPO yet.
- 2026-05-13 Subtask013 completed the switch: `g1_ppo_smoke` now defaults to
  `--asset-variant task023_hybrid`, generates
  `ankle_roll_hybrid_edge_boxes_no_points` inside each PPO run directory, and
  records `asset_resolution.json` plus the effective asset path in config and
  summary. `--asset-variant profile` keeps the old source-profile asset path.
  Local focused tests passed (`22 passed, 1 skipped`), local related caller
  tests passed (`27 passed, 4 skipped`), H200 focused tests passed
  (`23 passed`), and H200 real source-asset hybrid generation completed with
  `missing=[]`, `errors=[]`, `source_unchanged=true`, and `changed_geom_count=14`.
  No PPO training was run.
- 2026-05-13 Opened subtask014 after user asked to continue testing from the
  hybrid PPO switch. Ran diagnostic PPO smoke/gate only. H200 conservative
  5-update A/B showed hybrid improves reward and max training episode length
  over the source-profile asset (`94.17` versus `71.63`), but is not a
  long-horizon standing pass. H200 20-update hybrid standing gate improved the
  best training episode length to `100.75`, near but below the old task020
  2x-baseline threshold around `103.84`, and still ended with final tilt-reset
  sweeps. A conservative 20-update run, `legs`, and `legs_no_ankle_roll` action
  masks all reproduced the same late tilt reset. Hybrid zero-action no-update
  causality reproduced the same failure class with `first_tilt_chunk=3` and
  final `tilt_bad_count=1024`, so PPO updates are not required for the
  collapse.
- 2026-05-13 Opened subtask015 after user raised base-link placement at waist
  or hip. Source MJCF inspection confirmed the floating base/free joint is on
  `pelvis`, `waist_yaw_link` is a child of pelvis, the hip links are lower
  pelvis children, and the IMU site is on `torso_link`. H200 hybrid link trace
  over 130 zero-action steps showed pelvis and waist height are identical in
  the collapse window; torso is only slightly higher and hip links are lower.
  At step 123, before termination-height failure, `upright=0.2702` already
  triggers tilt reset while pelvis/waist/torso/hip z remain separated only by
  small offsets. Conclusion: changing the measured base frame to waist, torso,
  or hip would not remove the primary tilt failure.
- 2026-05-14 Opened subtask016 after user asked to inspect zero-action/hybrid
  step 80-130 COM, foot contact, base pitch/roll, and support polygon traces.
  Added `g1_zero_action_support_trace`, passed local focused tests
  (`28 passed, 4 skipped`) and H200 focused tests (`32 passed`). H200 trace
  `h200-gpu1-hybrid-step080-130-v1` found
  `first_com_outside_support_step=77`, `first_height_bad_step=120`,
  `first_tilt_step=123`, and `first_termination_height_bad_step=125`. The
  root pitch grows from `0.3357` rad at step 80 to `1.2948` rad at step 123,
  while COM signed margin drops from `-0.0153` at step 80 to `-0.6718` at step
  123. Foot support starts with both feet active, decays by step 116, becomes
  left-only at step 120, disappears at step 122, then tilt reset fires at
  step 123.

## Review

Status: diagnostic_not_passed; subtask012 evidence current.

- 2026-05-13 Final read-only reviewer found no blocking findings. Task023 is
  complete as a diagnostic result, not a passed standing-controller baseline.
  Decision: fixed stabilizer helps partially on the source asset, but
  asset/controller semantics still dominate and it is not ready to become the
  PPO standing baseline.
- 2026-05-13 Subtask004 confirms the next blocker more precisely: not a simple
  ankle-roll action mapping/sign bug, and not a pure bad-asset-only failure.
  The root cause is an unmatched Genesis contact/controller ABI: ankle-roll
  geoms carry the support contact, ankle-pitch contact remains 0.0, and clipped
  attitude targets can inject large early contact impulses on larger-spheres
  while source only sees comparable force after collapse. This originally
  suggested a contact-controller ABI probe; the user later constrained the next
  route to Genesis-only.
- 2026-05-13 User removed MuJoCo from the next route. Subtask005 narrowed the
  ABI issue further with Genesis-only probes: ankle-pitch action is the minimal
  trigger for the larger-spheres early impulse, but source does not reproduce
  it. Highest-value next route is Genesis-only asset/contact repair or audit
  around ankle-roll contact geometry under ankle-pitch targets. PPO standing
  remains closed.
- 2026-05-13 Subtask006 ruled out scalar larger-sphere radius tuning as the
  repair. A useful candidate must now alter contact support layout/body
  semantics, not only sphere size. PPO standing remains closed.
- 2026-05-13 Subtask007 identifies the root direction: make the foot/support
  collision geometry coherent first, then revisit standing control. The fixed
  controller's remaining 109-step failure on box support is a separate
  standing-dynamics blocker after early contact impulse is removed.
- 2026-05-13 Subtask008 turns that direction into a cleaner generated asset
  candidate and confirms the split: continuous sole contact can make early
  controller/contact forces sane, but a footprint-derived sole does not improve
  passive support or active standing horizon. PPO standing remains closed until
  a reviewed Genesis foot sole collision asset improves both early force and
  horizon gates.
- 2026-05-13 Subtask009 refines the root cause: Genesis standing depends on
  heel/toe edge support semantics in addition to a continuous midfoot contact.
  The old point supports are not merely noise; disabling them collapses passive
  horizon from 113 to 87 even with the same center box. The next root-solution
  attempt should replace points with explicit heel/toe edge boxes, not one
  large mesh-bbox sole slab.
- 2026-05-13 Subtask010 found that explicit heel/toe edge boxes only work when
  combined with the center pad: passive horizon reaches 116 with low early
  active force. Since active `attitude + all` still resets at 108, the next
  blocker is controller clipping/gain/target dynamics on this hybrid asset,
  not early contact geometry. PPO remains closed.
- 2026-05-13 Subtask011 identifies the controller pattern more precisely:
  continuous fixed attitude feedback is harmful, but a short early controller
  pulse followed by passive dynamics extends reset to 122. PPO remains closed
  because this is not sustained standing control; the next controller route
  should test decaying/gated control or inspect target/root traces around the
  40-80 step transition.
- 2026-05-13 Subtask012 answers the SONIC check: official SONIC assets and
  reference data look readable and the MuJoCo sim side starts, so this is not
  evidence of a corrupt SONIC asset. Full SONIC runability is not passed because
  the native deploy path fails before inference/control at TensorRT/CUDA
  runtime initialization. This does not certify the task023 Genesis asset as
  healthy; it only shows the official MuJoCo reference stack reaches a runtime
  environment blocker.
- 2026-05-13 Clarified result: official SONIC MuJoCo asset health is passed at
  the asset/sim-loop level. The previous "SONIC not passed" status applies only
  to the full native policy/control rerun after the CUDA12.9 TensorRT runtime
  extraction disappeared.
- 2026-05-13 Subtask013 makes the next PPO smoke use the best current hybrid
  foot asset by default while preserving source-profile fallback. The next
  task can now run a diagnostic PPO smoke without accidentally training on the
  original source contact asset.
- 2026-05-13 Subtask014 reopens PPO only as a diagnostic smoke/gate and does
  not pass standing. The hybrid asset improves the support horizon and reward
  quality, but zero-action hybrid still falls and 20-update PPO ends with the
  same finite-horizon tilt-reset class. Current blocker remains
  passive/contact/standing-dynamics semantics, not PPO plumbing, upper-body
  action, or direct ankle-roll action. A root fix should next target a more
  physically correct Genesis foot/support model or an explicit balance-control
  formulation that is evaluated against zero-action/passive horizons before
  retrying longer PPO.
- 2026-05-13 Subtask015 rejects base-link relabeling as a root fix. The
  physical MJCF root is already pelvis; waist is effectively colocated with it
  for the current height semantics, and torso/hip relabeling would mainly move
  scalar height thresholds while the tilt/upright failure remains. Do not
  rewrite the MJCF root to torso or hip without a separate asset-contract
  migration plan.
- 2026-05-14 Subtask016 identifies the finite-horizon passive failure pattern:
  forward pitch moves the COM projection outside the support polygon before
  reset, then support area and foot contact collapse, and only afterward do
  tilt and height termination fire. The next root-fix branch should target
  passive COM/support alignment or an explicit balance-control objective,
  not longer PPO or base-height threshold tuning.
