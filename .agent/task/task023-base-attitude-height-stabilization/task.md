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

## Review

Status: reviewed_no_blocking_diagnostic_not_passed.

- 2026-05-13 Final read-only reviewer found no blocking findings. Task023 is
  complete as a diagnostic result, not a passed standing-controller baseline.
  Decision: fixed stabilizer helps partially on the source asset, but
  asset/controller semantics still dominate and it is not ready to become the
  PPO standing baseline.
