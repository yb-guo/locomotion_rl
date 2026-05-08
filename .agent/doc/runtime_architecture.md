# Runtime Architecture

This document records the current train/inference architecture consensus.

## Train/Inference Unity

Train/inference unity means:

- same robot profile;
- same observation/action math;
- same policy/action semantics.

It does not mean training and deployment must share the exact same Python
`step()` function.

## Execution Shapes

Use two execution shapes over the same compiled robot profile.

Scalar path:

- single robot or single simulator;
- real robot deployment;
- smoke tests and GIF/video rollouts;
- strict finite checks, logging, watchdogs, and safety gates;
- may use dataclasses and clear backend boundaries.

Tensor path:

- vectorized training and large evaluation;
- no per-step YAML reads;
- no per-step joint-name lookup;
- no per-step dataclass allocation in the inner loop;
- use precompiled index arrays/tensors and batched math.

## Profile Ownership

Runtime configuration is loaded from YAML at initialization time.

Python dataclasses are schema/validation/runtime objects, not the authority for
SONIC G1 constants.

For the SONIC G1 body route, the authoritative control profile mirrors the
official SONIC deployment constants:

- 29DoF command/MuJoCo joint order;
- policy/IsaacLab order mapping;
- default angles in command order;
- action scales in command order;
- kp, kv, and force limits in command order.

The runtime must compile this YAML once into a fast profile object.

## Observation Builders

Low-level preprocessing should not force all policies into the same observation.

Each policy gets its own builder:

- `SONICObservationBuilder`;
- `PPOObservationBuilder`;
- `LocoFormerObservationBuilder`.

Builders may compute policy order, centered joints, gravity direction, token
layout, masks, and history windows themselves, but shared math helpers should be
used to avoid train/inference drift.

## Backend Boundary

Backends provide state and receive motor commands.

For scalar deployment/smoke:

```text
RobotBackend -> RobotState -> ObservationBuilder -> PolicyRuntime
  -> ActionBridge -> MotorCommand -> RobotBackend
```

For vectorized training:

```text
VectorizedBackend tensors -> TensorObservationBuilder -> Policy tensor
  -> TensorActionBridge -> target tensors -> VectorizedBackend
```

Both paths must use the same compiled profile and equivalent action/observation
math.

## LocoFormer Direction

The next phase remains G1-first. Interfaces should allow an `EmbodimentSpec`,
but task008 should not implement procedural multi-embodiment training yet.

Multi-configuration support is deferred until the G1 29DoF train/inference
profile foundation is stable.

## Task Discipline

Each implementation subtask must be a minimum closed loop:

- one narrow change;
- one local verification target;
- one recorded pass/fail result;
- no handoff at "code added but not exercised";
- H200 evidence when the subtask touches simulator/runtime behavior that only
  matters on the target.

Prefer many closed loops over one large refactor.

## Router Workflow

Use one task branch and one task worktree. The Router delegates implementation
to one coding subagent and read-only review to another subagent until the
subtask passes, then repeats this loop until the task is complete. The Router
records route, verification evidence, review result, branch/worktree state, and
remaining risk so outcomes stay explainable. `pr_gate.py` is not required for
task completion.
