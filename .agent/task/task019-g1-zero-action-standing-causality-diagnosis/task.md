# Task 019: G1 Zero-Action Standing Causality Diagnosis

## Goal

Find the first causal boundary for why the G1 27DoF no-hand Genesis standing
environment tilts and resets under `zero_action`.

Task018 showed that `zero_action` no-update already reproduces the reset wave,
so task019 must stay below PPO. The first work item is the highest-value
six-case control/pose gate.

This task is not a walking-quality or PPO-training claim.

## Scope

- Branch: `codex/task019-g1-zero-action-standing-causality-diagnosis`.
- Worktree:
  `../_worktrees/h200-locomotion-lab-task019-g1-zero-action-standing-causality-diagnosis`.
- Base: stacked on task018 commit `cfea512`.
- Add a standalone zero-action standing causality probe.
- Run the first six H200 gate experiments:
  - `current_pose + root_z=1.20 + genesis_position`;
  - `current_pose + root_z=1.20 + genesis_position_resend_physics`;
  - `current_pose + root_z=1.20 + custom_pd_torque`;
  - `unitree_gym_pose + root_z=0.80 + genesis_position`;
  - `unitree_gym_pose + root_z=0.80 + genesis_position_resend_physics`;
  - `unitree_gym_pose + root_z=0.80 + custom_pd_torque`.

`current_pose` means the task018 failing operational baseline:
`tall_crouch` legs with hip pitch `-0.06`, knee `0.12`, ankle pitch `-0.07`
and root z `1.20`.

## Non-Goals

- No PPO runs.
- No LocoFormer integration.
- No SONIC integration.
- No ONNX export.
- No rendering/GIF/video.
- No dataset/checkpoint/asset/upstream download.
- No writes/deletes under `/mnt/workspace` or `/mnt/workspace1`.
- No change to `GenesisG1SceneBackend`.
- Do not mark passed without local tests, H200 focused tests, six-case H200
  evidence, and read-only review.

## H200 Protocol

Remote commands must use:

```bash
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'
```

All remote code, outputs, and intermediate files must stay under:

```text
/root/agent_workspace/project
```

Default GPU:

```text
CUDA_VISIBLE_DEVICES=1
physical_gpu=1
logical_cuda_device=cuda:0
```

Remote project:

```text
/root/agent_workspace/project/h200-locomotion-lab-task019-g1-zero-action-standing-causality-diagnosis
```

Run dir:

```text
outputs/task019/zero_action_standing_causality/<run_id>/
```

## Diagnose Loop

### Feedback Loop

Use a deterministic zero-action standing probe:

- seed `0`;
- `n_envs=1024`;
- `50 chunks`;
- `32 policy steps per chunk`;
- total `1600` policy steps;
- standing commands only;
- G1 27DoF no-hand Genesis asset;
- no policy model;
- no PPO update;
- no reward/value/advantage logic.

Pass signal:

```text
reset_count=0 and tilt_bad_count=0 for all chunks
```

Fail signal:

```text
first_tilt_chunk/step, reset_count, tilt_bad_count, and reset cause
```

Record per chunk:

- reset/tilt/height counters;
- root height mean/min;
- upright mean/min;
- joint position error rms/max;
- joint velocity rms/max;
- control force or custom torque rms/max;
- force saturation ratio;
- foot/body contact counters when available;
- throughput.

### Ranked Hypotheses

1. **Genesis position-control calling semantics are unstable**
   - Prediction: resending the same position target every physics substep or
     switching to custom torque PD makes the zero-action reset wave disappear
     or move much later.
2. **Current reset/default pose is not a stable standing equilibrium**
   - Prediction: `unitree_gym_pose` improves or eliminates zero-action resets
     under the same control mode.
3. **PD gain or force limit profile is not suitable for Genesis**
   - Prediction: named Unitree-style gain/force profiles improve stability
     after the best control/pose candidate is found.
4. **Reset/contact state is dirty or the asset/contact dynamics are unstable**
   - Prediction: velocity hard-zero, settle-before-eval, root-z sweep, or
     contact instrumentation changes the failure timing.

## Stop Rules

- Run the six-case control/pose gate before any gain/force sweep.
- If any six-case gate candidate passes 1600 zero-action policy steps, stop the
  broad matrix and verify around that candidate.
- If `custom_pd_torque` passes while Genesis position-control modes fail,
  classify the next area as Genesis position-control semantics/call path.
- If `unitree_gym_pose` passes while `current_pose` fails under the same
  control mode, classify the next area as reset/default pose.
- If all six gate cases fail, continue only to targeted gain/force or
  reset/contact diagnostics. Do not run PPO.
- Change one variable per probe except for the deliberate six-case gate axes
  declared in subtask 003.

## Acceptance

- Router creates task/subtask docs before coding.
- Coding subagent implements standalone probe tooling.
- Read-only reviewer reviews boundary, correctness, and evidence.
- Local focused tests pass.
- Local full pytest passes or task-scoped tests plus documented reason.
- H200 focused tests pass.
- H200 six-case gate evidence is recorded.
- Decision states whether the next area is Genesis control semantics,
  reset/default pose, gain/force profile, or reset/contact/asset dynamics.

# Route

1. `001-repro-contract.md`
2. `002-probe-tool-and-instrumentation.md`
3. `003-h200-six-case-control-pose-gate.md`
4. `004-targeted-gain-force-followup.md`
5. `005-reset-contact-state-followup.md`
6. `006-review-and-decision.md`

# Log

- 2026-05-11 Created task019 branch/worktree from task018 commit `cfea512`.
- 2026-05-11 Router created task/subtask route before coding.
- 2026-05-11 Coding subagent implemented standalone zero-action causality
  probe and focused tests.
- 2026-05-11 Pre-H200 read-only reviewer found no blocking findings.
- 2026-05-11 Local full pytest passed: `229 passed, 12 skipped`.
- 2026-05-11 H200 focused tests passed: `38 passed in 1.67s`.
- 2026-05-11 H200 six-case control/pose gate completed; all six cases failed.
- 2026-05-11 First gate decision: target resend and custom torque PD do not
  rescue the failure; `unitree_gym` pose fails earlier than current.
- 2026-05-11 Final read-only reviewer found no blocking findings for the
  completed first gate and agreed task019 should remain in progress.
- 2026-05-11 Targeted gain-force follow-up completed on H200. No profile
  strictly passed; `unitree_leg_gains`, `global_kv_4x`,
  `global_kp_0_5x_kv_2x`, and `knee_ankle_kp_2x_kv_2x` recovered to final
  zero-reset chunks after an early all-env tilt/reset wave.
- 2026-05-11 `force_limit_2x` matched baseline with zero force saturation, so
  force limit is not supported as the primary cause.
- 2026-05-11 Reset/contact follow-up completed on H200. Warmup-only,
  full pre-eval reset, selected-env pre-eval reset, current-pose root-z sweep,
  and `unitree_gym` pose root-z sweep all failed with `max_reset_count=1024`.
- 2026-05-11 Representative metrics show periodic real falls every three
  chunks under `unitree_leg_gains`, not a one-time reset artifact.
- 2026-05-11 Final read-only reviewer found no blocking findings and confirmed
  task019 should not be marked passed.

# Review

Status: complete with no pass.

- Do not mark task019 passed yet.
- Completed scope: probe implementation, local/H200 focused tests, six H200
  gate experiments, targeted H200 gain-force matrix, and H200 reset/contact
  follow-up.
- Final decision: task019 found a causal boundary below PPO. The current G1
  27DoF no-hand Genesis standing setup has a periodic real zero-action fall
  every roughly three chunks. Simple control mode changes, Unitree-style gains,
  force-limit scaling, warmup/reset semantics, reset root-z sweeps, and the
  existing `unitree_gym` pose do not produce a strict 1600-step standing pass.
- Next boundary: create a standing-pose micro-sweep or inspect
  asset/contact/inertia details before any PPO run.
