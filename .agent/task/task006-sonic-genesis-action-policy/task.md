# Task 006: SONIC Genesis Action Replay Then Policy

## Goal

Move from SONIC reference joint-position replay to true action-driven Genesis G1
rollout.

The task order is fixed:

1. L1 action replay: drive the validated Genesis 29-motor G1 env with explicit
   29D normalized action sequences through `GenesisG1Env.step(action)`.
2. L2 SONIC policy: connect the real SONIC policy forward path only after L1
   passes with evidence.

Do not continue to L2 unless L1 passes on H200.

## Scope

- Genesis G1 29-motor action replay harness.
- Action CSV or deterministic action fixture loading.
- H200 smoke metrics:
  - finite state;
  - base height range;
  - min link height where available;
  - action range;
  - max qvel;
  - optional GIF/contact sheet.
- SONIC policy I/O inspection and rollout after action replay passes.

## Non-Goals

- No training loop yet.
- No PPO baseline yet.
- No Isaac Lab route.
- No new robot assets, checkpoints, datasets, or upstream repo downloads unless
  explicitly approved.

## Subtasks

- `001-genesis-action-replay.md`
- `002-sonic-policy-rollout.md`

## Current Known Inputs

- H200 run root:
  `/root/h200-locomotion-lab-runs/task004-genesis-g1-baseline`
- Valid Genesis asset:
  `/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml`
- Valid Genesis motor DOF order:
  `(6, 9, 12, 15, 19, 23, 7, 10, 13, 16, 20, 24, 8, 11, 14, 17, 21, 25, 27, 29, 31, 33, 18, 22, 26, 28, 30, 32, 34)`
- Existing reference replay evidence:
  `.agent/task/task004-genesis-g1-baseline/002-genesis-env-reset-step.md`

## Review

Status: L1 action replay passed; L2 SONIC policy rollout remains open.

L1 pass evidence is recorded in `001-genesis-action-replay.md`. L2 may now
inspect and connect the SONIC policy path, but must not erase the distinction
between action replay and real policy rollout.
