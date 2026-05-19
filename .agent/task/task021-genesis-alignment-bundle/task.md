# Task 021: Genesis Alignment Bundle

## Goal

Extract the parameters that define the G1 standing/training semantics and
compare the SONIC/MJCF sources against the current Genesis training env.

The output must make it obvious which values are already aligned, which values
are only configurable at runtime, and which values are still hidden inside the
asset or Genesis defaults.

This task is a diagnosis/instrumentation task, not a PPO-training task.

## Scope

- Branch: `codex/task021-genesis-alignment-bundle`.
- Worktree:
  `../_worktrees/h200-locomotion-lab-task021-genesis-alignment-bundle`.
- Base: `master` after task019 merge.
- Add a standalone alignment probe/report tool.
- Use existing prepared assets only.
- Compare at least:
  - default joint qpos/default angles;
  - reset/root pose defaults;
  - PD gains and force limits;
  - control timestep/decimation/policy rate;
  - action scale;
  - MJCF contact/friction/solver fields when present;
  - Genesis training backend fields that are actually applied.

## Non-Goals

- No PPO runs.
- No LocoFormer.
- No ONNX.
- No rendering/GIF/video.
- No new asset/importer.
- No Menagerie or `scene_mjx.xml` importer work.
- No dataset/checkpoint/upstream repo download.
- No writes/deletes under `/mnt/workspace` or `/mnt/workspace1`.
- No change to `GenesisG1SceneBackend`.
- Do not mark passed without local tests, report evidence, and read-only review.

## H200 Protocol

Remote commands must use:

```bash
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'
```

All remote code, outputs, and intermediate files must stay under:

```text
/root/agent_workspace/project
```

Default GPU metadata if a Genesis runtime probe is run:

```text
CUDA_VISIBLE_DEVICES=1
physical_gpu=1
logical_cuda_device=cuda:0
```

Remote project:

```text
/root/agent_workspace/project/h200-locomotion-lab-task021-genesis-alignment-bundle
```

Default output root:

```text
outputs/task021/genesis_alignment_bundle/
```

## Diagnose Loop

### Feedback Loop

Build a deterministic local tool that emits one JSON object with:

- source profile paths and asset paths;
- 29DoF SONIC profile bundle;
- 27DoF Genesis training profile bundle;
- mapped 29DoF-to-27DoF comparison;
- root/reset pose values used by planner defaults and backend defaults;
- parsed MJCF compiler/option/default/geom/contact fields when the XML is
  available;
- explicit `missing` entries for values not represented in current training
  config.

Pass signal:

```text
tool exits 0, JSON schema is stable, mapped control arrays match expected
27DoF values, missing contact/solver values are explicitly reported rather
than silently ignored
```

Fail signal:

```text
tool cannot run locally without Genesis, omits a requested parameter family,
or claims alignment for values that are not actually represented
```

### Ranked Hypotheses

1. **Control profile is mostly aligned, but root/reset semantics are not**
   - Prediction: default angles, gains, force limits, and timing compare cleanly,
     while backend reset root differs from planner/root evidence.
2. **Action scale is double-configurable and needs explicit reporting**
   - Prediction: profile action scales are present, but runtime
     `action_scale_mult` can change the applied training scale.
3. **Contact/friction/solver semantics are the missing boundary**
   - Prediction: current profile/backend has no structured contact/solver
     config, and any values found come only from MJCF/XML or old env notes.
4. **Asset availability changes what can be proven locally**
   - Prediction: local runs can still compare profile/backend config, but full
     MJCF contact/solver extraction requires the prepared H200 asset path.

## Stop Rules

- If the report cannot run without importing Genesis, stop and fix the local
  feedback loop first.
- If a requested parameter family is unavailable, report it as `missing` with a
  source reason; do not infer values.
- If mapped 29DoF-to-27DoF control arrays do not match the 27DoF profile, stop
  before H200 runtime probing.
- If read-only review finds a blocking evidence or correctness issue, fix and
  rerun review before marking the subtask complete.

## Route

1. `000-contract-and-feedback-loop.md`
2. `001-bundle-extractor-and-alignment-report.md`
3. `002-h200-asset-report-evidence.md`
4. `003-review-and-decision.md`
5. `004-backend-contact-solver-boundary.md`
6. `005-rigid-options-standing-ablation.md`
7. `006-standing-semantics-root-cause-matrix.md`
8. `007-failure-onset-trace-and-combo-confirmation.md`
9. `008-ankle-foot-asset-contact-audit.md`

## Acceptance

- Router creates task/subtask docs before coding.
- Coding subagent implements scoped code changes.
- Read-only reviewer reviews boundary, correctness, and evidence.
- Local focused tests pass.
- The report is emitted and recorded as evidence.
- H200 asset report is recorded if the prepared asset is available.
- Decision states one of:
  - alignment bundle complete and standing PPO can consume the report;
  - control/root alignment gap found;
  - contact/friction/solver gap found and needs a follow-up asset/backend task.

## Log

- 2026-05-12 Created after task019 and task020 diagnostics showed the standing
  failure must be separated from PPO updates and inspected at the environment
  semantics boundary.
- 2026-05-12 Subtask001 local feedback loop implemented. Focused local command:
  `PYTHONPATH=src python -m pytest tests/test_g1_genesis_alignment_bundle.py
  tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q
  -p no:cacheprovider` -> 34 passed. Local report:
  `outputs/task021/genesis_alignment_bundle/local_profile_report.json`;
  alignment status `pass`, mapped control match `true`, Genesis timing
  self-consistent (`sim_dt_s=0.005`, `decimation=4`, `policy_rate_hz=50`),
  XML asset present `false`, missing count `13`. Missing records include
  `genesis_27dof_training_profile.contact_friction_solver_config` and
  `vectorized_genesis_backend.contact_friction_solver_config`, so contact/
  solver semantics are not represented in the current training profile/backend
  config. Missing records also include SONIC profile timing fields and MJCF
  decimation/policy-rate fields because they are not represented in those
  sources.
- 2026-05-12 Subtask002 H200 asset report completed through guarded commands.
  H200 focused tests passed 34/34. H200 report
  `outputs/task021/genesis_alignment_bundle/h200_asset_report.json`:
  `mapped_control_match=true`, `xml_asset_present=true`, `missing_count=11`;
  Genesis timing self-consistent (`0.005 * 4 -> 50 Hz`); MJCF compiler has
  `angle=radian`, `meshdir=meshes`; MJCF default groups expose joint
  `armature/damping/frictionloss`; MJCF lacks explicit `<option>`, `<contact>`,
  and geom-level contact/friction/solver fields.
- 2026-05-12 Final read-only review passed with no blocking findings. Decision:
  control profile and timing reporting are now explicit enough for diagnosis,
  but contact/friction/solver semantics remain a real profile/backend gap.
- 2026-05-12 User chose to continue inside task021. Subtask004 added an
  optional backend rigid contact/solver config boundary. Router reran expanded
  local related tests:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_vectorized_genesis_backend.py tests/test_g1_genesis_alignment_bundle.py tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q -p no:cacheprovider`
  -> 49 passed.
- 2026-05-12 Subtask004 H200 verification found and fixed the Genesis
  `constraint_solver` enum boundary: strings are mapped to
  `gs.constraint_solver.Newton`/`CG` before constructing `RigidOptions`.
  Expanded local related tests now pass 51/51. H200 guarded related tests pass
  51/51. H200 real Genesis enum probe used `CUDA_VISIBLE_DEVICES=1`
  (`physical_gpu=1`, `logical_cuda_device=cuda:0`) and built a scene with
  `applied_rigid_options=True`, `missing=[]`, `SCENE_BUILT 1 27 27`.
  Regenerated H200 report remains `status=pass`, `mapped_control_match=true`,
  `xml_asset_present=true`, `missing_count=11`.
- 2026-05-12 User requested continuing inside task021 and running the next
  diagnosis. Subtask005 created to ablate the newly exposed
  `rigid_contact_solver` boundary against zero-action standing metrics.
- 2026-05-12 Subtask005 H200 ablation completed as one scenario per process
  after multi-scenario SSH sessions proved unstable. All runs used
  `CUDA_VISIBLE_DEVICES=1`, `physical_gpu=1`, `logical_cuda_device=cuda:0`,
  `n_envs=512`, `chunks=8`, `chunk_steps=32`. `default_unset`,
  `newton_solver_only`, and `newton_mujoco_contact` all failed with
  `first_tilt_step=64`, `max_reset_count=512`, `max_tilt_bad_count=512`, and
  near-identical `max_contact_force` around 173. `newton_solver_bundle` was
  worse, ending with `final_reset_count=512`, `final_tilt_bad_count=512`, and
  `final_root_height_min=0.354421`. Local and H200 focused tests pass 57/57.
- 2026-05-12 Subtask006 created to run a bounded control/gain/pose/root-height
  standing semantics matrix using the existing zero-action standing probe as
  one process per scenario.
- 2026-05-12 Subtask006 H200 standing semantics matrix completed. Local
  focused tests pass 27/27; local expanded tests pass 64/64 after review fixes;
  H200 focused tests with the same expanded command pass 64/64. All H200 runs used
  `CUDA_VISIBLE_DEVICES=1`, `physical_gpu=1`, `logical_cuda_device=cuda:0`,
  `n_envs=512`, `chunks=8`, `chunk_steps=32`. Control layer showed
  `baseline_current` and `control_resend_physics` are identical
  (`first_tilt_step=64`, `max_reset=512`, `final_root_height_min=0.636994`,
  `final_upright_mean=0.934044`); `control_custom_pd_torque` improved final
  height/upright (`0.681814`, `0.955294`) but not first failure step. Gain
  layer showed `gain_unitree_leg` is the best single improvement
  (`final_root_height_min=0.742497`, `final_upright_mean=0.978485`) but still
  fails at `first_tilt_step=64`, `max_reset=512`. Pose/root layer showed
  `pose_unitree_gym` is worse (`first_tilt_step=32`, `final_reset=512`) and
  raising `root_z` to `0.90`, `1.00`, or `1.10` does not remove/delay the
  failure. Remote summaries:
  `outputs/task021/standing_semantics_matrix/h200-subtask006-control-20260512-01/summary.json`,
  `outputs/task021/standing_semantics_matrix/h200-subtask006-gain-20260512-01/summary.json`,
  `outputs/task021/standing_semantics_matrix/h200-subtask006-pose-root-20260512-01/summary.json`.
- 2026-05-12 Subtask006 read-only review found a blocking matrix robustness
  issue: scenario exceptions could abort the matrix. Fixed per-scenario error
  capture/continue behavior and error ranking; copied H200 summary artifacts
  into local `outputs/task021/standing_semantics_matrix/`.
- 2026-05-13 Subtask007 created to inspect the failure onset at 1-policy-step
  granularity and confirm whether combining the best subtask006 control/gain
  changes delays the step-64 failure.
- 2026-05-13 Subtask007 H200 trace completed. Local focused tests pass 35/35;
  local expanded tests pass 70/70; H200 expanded tests pass 70/70. All H200
  traces used `CUDA_VISIBLE_DEVICES=1`, `physical_gpu=1`,
  `logical_cuda_device=cuda:0`, `n_envs=512`, and `chunk_steps=1`.
  One-step onset shows subtask006's `first_tilt_step=64` was a coarse lower
  bound from `chunk_steps=32`: baseline first tilt is step 88. The combined
  `custom_pd_torque + unitree_leg_gains` scenario delayed onset to step 93 and
  repeated step 93 in a confirm rerun, but still failed with `max_reset=512`.
  `force_saturation_ratio=0.0`; baseline/custom-PD onset top joint errors are
  ankle pitch, while Unitree gains reduce ankle pitch error but root/upright
  still collapse with contact spikes.
- 2026-05-13 Subtask008 created to audit ankle/foot asset contact semantics and
  collect link-level z/contact-force evidence around the one-step failure onset.
- 2026-05-13 Subtask008 H200 audit completed. Local focused tests pass 6/6;
  local expanded tests pass 76/76; H200 expanded tests pass 76/76. XML audit
  shows ankle-roll links have `mass=0.608`, one non-colliding visual mesh
  (`contype=0 conaffinity=0`), plus four `size=0.005` contact geoms without
  explicit `friction/condim/solref/solimp/priority`; this matches the Genesis
  dubious-mass warning for `left/right_ankle_roll_link`. Link-level traces show
  contact force occurs on ankle-roll links only: ankle-pitch links have
  `contact_force_max=0`; baseline ankle-roll force max is about `173` with
  first tilt at step 89, and combo force max is about `406` with first tilt at
  step 94.

## Review

Status: passed. Subtask008 read-only review found no blocking findings.

Decision: control profile, timing, and the backend rigid contact/solver
configuration boundary are now explicit. H200 rigid-options and standing
semantics matrices point away from isolated control-call frequency, simple
gain/force-limit changes, and root-z/Unitree-pose fixes. H200 onset tracing
shows combined `custom_pd_torque + unitree_leg_gains` reproducibly delays the
failure from step 88 to step 93 but does not solve standing. Remaining likely
causes are asset inertial/contact geometry around the ankle/foot chain or a
missing active base-attitude/height stabilizer, not PPO reward tuning alone.
Subtask008 now localizes the immediate contact path to `left/right_ankle_roll_link`.
