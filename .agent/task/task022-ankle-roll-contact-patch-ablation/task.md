# Task 022: Ankle Roll Contact Patch Ablation

## Goal

Test whether a controlled project-local G1 XML contact patch improves the
standing failure localized by task021 to `left/right_ankle_roll_link`.

This task is still diagnosis-first. The pass/fail signal is zero-action standing
and link-level contact trace, not PPO reward.

## Scope

- Branch: `codex/task022-ankle-roll-contact-patch-ablation`.
- Worktree:
  `../_worktrees/h200-locomotion-lab-task022-ankle-roll-contact-patch-ablation`.
- Base: task021 diagnostic code and evidence.
- Remote project:
  `/root/agent_workspace/project/h200-locomotion-lab-task022-ankle-roll-contact-patch-ablation`.
- Use the prepared G1 asset as input only:
  `/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_27dof_nohand.xml`.
- Generate patched XML variants under the task project, for example:
  `outputs/task022/ankle_roll_contact_patch/assets/`.
- Run zero-action standing and link-level traces against patched XML variants.
- Compare against the task021 baseline signatures:
  - baseline first tilt around step 88-89;
  - combo first tilt around step 93-94;
  - contact force localized to ankle-roll links.

## Non-Goals

- No PPO.
- No LocoFormer.
- No rendering/GIF/video.
- No ONNX.
- No downloads of assets, datasets, checkpoints, or upstream repos.
- No edits to the prepared source asset in `/root/h200-locomotion-lab-runs`.
- No writes/deletes under `/mnt/workspace` or `/mnt/workspace1`.
- No change to `GenesisG1SceneBackend`.
- Do not claim success if a patch only changes reset semantics while the robot
  still physically collapses.

## H200 Protocol

Remote commands must use:

```bash
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'
```

All remote code, generated XML variants, outputs, and intermediate files must
stay under:

```text
/root/agent_workspace/project
```

Default GPU metadata for Genesis runtime probes:

```text
CUDA_VISIBLE_DEVICES=1
physical_gpu=1
logical_cuda_device=cuda:0
```

## Diagnosis Context

Task021 evidence points to the ankle-roll contact path:

- `left/right_ankle_roll_link` mass is `0.608kg`.
- Genesis warns that these ankle-roll link masses are dubious compared with
  geometry.
- The prepared XML has one non-colliding visual mesh
  (`contype=0 conaffinity=0`) plus four direct `size=0.005` point-like contact
  geoms under each ankle-roll link.
- There are no explicit ankle/foot `friction`, `condim`, `solref`, `solimp`,
  or `priority` fields.
- Link trace shows ankle-pitch contact force is zero, while ankle-roll link
  contact force appears before tilt/reset.

## Diagnose Loop

### Feedback Loop

Build a deterministic patch generator/report tool that:

- copies the prepared G1 XML into project-local output paths;
- creates a small set of named patch variants;
- records an XML diff summary and exact target-body geom/inertial changes;
- emits patched asset paths;
- runs, or prepares commands to run, the existing zero-action/link trace probes
  against each patched asset.

Primary H200 pass signal:

```text
patched XML variant delays or removes first tilt/reset relative to both baseline
and combo task021 signatures, while root height/upright and ankle-roll contact
force improve rather than merely hiding termination
```

Failure signal:

```text
first tilt remains around step 89 baseline / step 94 combo, ankle-roll force
remains localized and large, or patch causes earlier height/upright collapse
```

### Ranked Hypotheses

1. **Ankle-roll contact support is too point-like**
   - Prediction: replacing or augmenting the four `size=0.005` geoms with a
     larger foot-support collision shape will delay first tilt and reduce
     ankle-roll contact force spikes.
2. **Missing contact/friction attrs are the unstable boundary**
   - Prediction: adding explicit `friction`/`condim` to ankle-roll support
     geoms improves contact stability without changing inertial mass.
3. **Mass/contact geometry mismatch is the issue**
   - Prediction: changing only contact shape/friction helps partially, but a
     mass/inertial patch or better upstream asset is needed to remove failure.
4. **Contact geometry is not sufficient**
   - Prediction: all controlled contact patches leave first tilt unchanged,
     pushing the next task toward active base attitude/height stabilization.

## Stop Rules

- First implement patch generation and static XML audit; do not run H200 before
  the generated XML is inspectable and local tests pass.
- Start with at most three patch variants:
  - `ankle_roll_friction_attrs`;
  - `ankle_roll_larger_spheres`;
  - `ankle_roll_box_support`.
- Run baseline source asset and one patch family at a time.
- If no patch delays first tilt by at least 10 policy steps in zero-action
  standing, stop and record contact patch as insufficient.
- If a patch improves first tilt, rerun the same patch once before treating it
  as evidence.
- Do not proceed to PPO until zero-action/link-trace evidence and read-only
  review both pass.

## Route

1. `000-contract-and-feedback-loop.md`
2. `001-project-local-xml-patch-generator.md`
3. `002-h200-zero-action-patch-ablation.md`
4. `003-review-and-decision.md`

## Acceptance

- Router creates task/subtask docs before coding.
- Coding subagent implements scoped code changes.
- Read-only reviewer reviews patch correctness and evidence.
- Local focused tests pass.
- H200 generated XML variants stay under `/root/agent_workspace/project`.
- H200 zero-action/link-trace evidence compares source asset against patch
  variants with `CUDA_VISIBLE_DEVICES=1`.
- Decision states one of:
  - contact patch improves standing enough to become a candidate training asset;
  - contact patch helps partially but active stabilizer is still needed;
  - contact patch is insufficient and the next task should target controller or
    upstream asset semantics.

## Log

- 2026-05-13 Created after task021 localized the immediate contact path to
  `left/right_ankle_roll_link` and ruled out PPO/reward/reset as the first
  thing to change.
- 2026-05-13 Task022 branch/worktree created from task021 dependency baseline.
  Task021 is not merged yet, so this branch intentionally carries task021
  diagnostic tools as a dependency baseline.
- 2026-05-13 Subtask001 implemented a project-local XML patch generator for
  `left/right_ankle_roll_link` variants and local fixture tests. Focused local
  verification passed: `PYTHONPATH=src python -m pytest
  tests/test_g1_ankle_roll_contact_patch.py -p no:cacheprovider`.
- 2026-05-13 Subtask001 read-only review passed with no blocking findings.
  Remaining risk is intentionally narrow support-geom matching; H200 ablation
  must inspect generated real-asset XML before runtime comparison.
- 2026-05-13 Subtask002 local preparation added generated-asset override
  plumbing to zero-action, failure-onset, and contact-audit link trace tools.
  No H200 run performed. Initial review found a blocking compatibility
  regression in `build_run_config`; it was fixed and re-reviewed. Router
  verification passed:
  `PYTHONPATH=src python -m pytest
  tests/test_g1_rigid_options_standing_ablation.py
  tests/test_g1_zero_action_standing_causality.py
  tests/test_g1_failure_onset_trace.py
  tests/test_g1_ankle_foot_asset_contact_audit.py -q -p no:cacheprovider`
  -> 37 passed, 4 skipped.
- 2026-05-13 H200 patch import hit a generator blocker before ablation:
  Genesis tried to open generated-XML-relative
  `assets/meshes/right_ankle_roll_link.STL` because the source MJCF used a
  relative `compiler.meshdir` and the generated XML moved away from the source
  XML directory. Local fix updates `g1_ankle_roll_contact_patch` to rewrite an
  existing relative `compiler meshdir` to the absolute source-resolved meshdir
  in generated variants, while preserving the source XML and leaving absent
  compiler tags absent. Summary evidence now records source meshdir, resolved
  source meshdir, output meshdir, and whether rewrite happened. Focused local
  verification passed: `PYTHONPATH=src python -m pytest
  tests/test_g1_ankle_roll_contact_patch.py -q -p no:cacheprovider` -> 5
  passed. No H200 rerun performed.
- 2026-05-13 H200 deploy `36807e4` passed focused remote tests through
  `run_guarded.sh` with `CUDA_VISIBLE_DEVICES=1` -> 46 passed. Patch generator
  run `h200-gpu1-patches-v2` produced all three variants under
  `/root/agent_workspace/project/h200-locomotion-lab-task022-ankle-roll-contact-patch-ablation/outputs/task022/ankle_roll_contact_patch/`,
  with `source_unchanged=true`, `missing=[]`, `errors=[]`, and source-resolved
  absolute `compiler.meshdir`.
- 2026-05-13 H200 zero-action/link-trace evidence collected on physical GPU 1.
  Source first tilt reproduced at step 88. `ankle_roll_friction_attrs` stayed
  at step 88. `ankle_roll_larger_spheres` delayed first tilt to step 106 and
  reproduced step 106 in confirm. `ankle_roll_box_support` delayed first tilt
  to step 113 and reproduced step 113 in confirm. Link-level traces kept
  ankle-pitch contact force at 0.0; source/friction ankle-roll force max was
  about 294, larger-spheres about 241, and box-support about 639-663. All
  variants still failed from tilt/reset, so this task remains diagnosis, not a
  PPO pass.
- 2026-05-13 Final read-only review found no blocking findings. Decision:
  contact patch helps partially but is insufficient for stable passive
  standing. Keep `ankle_roll_larger_spheres` as the cleanest controlled
  comparison asset for follow-up; do not treat `ankle_roll_box_support` as a
  training asset without separate high-contact-force geometry review. PPO stays
  closed.

## Review

Status: passed with no blocking findings. Task022 decision is partial-help,
not a standing/PPO pass.
