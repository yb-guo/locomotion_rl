# 001: Env Contract And Randomization Scope

## Route

Define the first task028 environment contract narrowly enough that it can be
trained with the existing MJLab/RSL-RL PPO MLP stack.

Decisions for the first pass:

1. Fixed topology.
2. Fixed DoF count.
3. Fixed observation tensor shape.
4. Fixed action tensor shape.
5. Randomize dynamics and morphology parameters inside that fixed contract.
6. Defer LocoFormer-style variable topology, padding/masks, morphology tokens,
   and action-token decoding until after the MLP baseline passes.
7. Use a G1-like whole-body robot with simplified grippers.
8. Do not use full dexterous hands in the first pass.

Initial robot contract:

- Base: floating-base humanoid, G1-like proportions.
- Lower body: G1-like legs and feet.
- Waist: G1-like waist joints if available in the base asset.
- Upper body: G1-like shoulders, elbows, and wrists.
- End effectors: simplified parallel grippers.
- Topology: fixed for all training episodes.
- Action space: fixed-size joint position target action, including two total
  gripper action dimensions.
- Observation space: fixed-size flat observation for the first MLP baseline.
- Gripper contact: no object contact task and no required ground contact in the
  first pass. Keep gripper collision conservative or disabled.

Initial randomization candidates:

- link mass scale
- body COM offset
- body inertia scale
- motor strength scale
- PD gain scale
- joint damping/friction
- contact friction
- encoder bias
- observation noise
- action delay / action smoothing
- pushes

Initial eval gates:

- deterministic no-randomization eval
- fixed `vx=0.5, vy=0, yaw=0` walking
- randomized holdout eval
- render evidence from a saved checkpoint

## Minimal Closed Loop

Feedback loop:

1. Write a machine-readable contract summary for the first-pass task:
   task id, action terms, expected action dims, observation terms, randomized
   fields, disabled/deferred features, and eval gates.
2. Run a lightweight contract checker that validates the summary against the
   implemented config once 002 exists.

Pass:

- Contract states fixed topology and fixed DoF.
- Contract states action order as `[body29, gripper_left, gripper_right]`.
- Contract states gripper contact/object tasks are deferred.
- Contract defines deterministic eval, randomized holdout eval, and render
  evidence paths.
- No policy architecture change is required by the contract.

Fail:

- Any variable topology, variable DoF, padding/mask, or token action decoder is
  required in the first pass.
- Gripper action order is ambiguous.
- Randomization scope cannot be toggled off for a deterministic control eval.

Evidence:

- This subtask doc plus the eventual config inspection output from 002.

## Log

- 2026-05-19 Opened after the user confirmed not to support variable topology
  or variable DoF in the first pass.
- 2026-05-19 Contract validated by the 002 implementation smoke. The active
  first-pass task is `Unitree-G1-Gripper-Flat`; action order is fixed as
  `body_joint_pos` 29 dims followed by `gripper_joint_pos` 2 dims, total
  action dim 31. Actor observation shape is 104 and critic observation shape
  is 119 in the flat task smoke.

## Review

Status: passed for the first-pass fixed-topology contract.

The pass condition for this subtask was a written env contract precise enough
to implement without changing policy architecture. The 002 smoke confirms the
contract is implementable with the existing MJLab/RSL-RL MLP action interface:
fixed topology, fixed action order `[body29, gripper_left, gripper_right]`,
no variable-DoF padding/masks, and no gripper object-contact task.
