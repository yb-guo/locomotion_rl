# 002 Unified G1-like Slot Contract

## Route

Create the fixed semantic action/observation slot contract for G1-like variants.

The unified action space is slot-semantic:

```text
slot 0 = left_hip_pitch
slot 1 = left_hip_roll
slot 2 = left_hip_yaw
...
```

Slots must never be reused for different meanings. A robot variant maps its
actual joint names into the fixed slots through a selector and mask.

First scope:

- Fixed 29-slot body-only G1/SONIC `command_mujoco` contract covering lower
  body, waist, and arms.
- Gripper, hand, and finger slots do not enter the current action dimension.
  Any later G1-like + gripper scope must be an explicit extension that
  redefines the action dimension and compatibility mapping.
- Keep the current G1 action contract available as a compatibility mapping.
- Actor receives masked unified observations/actions; exact morphology
  parameters remain debug/critic-only unless a later baseline opens them.

## Minimal Closed Loop

Close this slice locally with a schema/mapping test that constructs a toy
G1-like variant and proves:

- slot names are unique and stable;
- action dimension is fixed even when a variant masks optional slots;
- present slots select the expected action values;
- missing slots are masked and cannot be controlled accidentally;
- the current G1 mapping round-trips existing MJLab joint/action order.

## Evidence Gate

Reviewable evidence is a small schema file or JSON fixture plus local command
output showing the checks above. H200 simulation is not required for this slice,
but later morphology and eval slices must consume this schema instead of
hard-coding joint order.

Acceptance:

- A machine-readable slot schema exists.
- A G1 mapping exists and round-trips existing MJLab joint/action order.
- Local tests prove missing slots are masked, present slots select the right
  action values, and slot names are unique.
- No morphology generator depends on ad hoc joint-order assumptions outside this
  schema.

## Subagent Ownership

- Worker owns the slot contract doc, any small machine-readable slot schema, and
  local schema tests if implementation is later authorized.
- Worker does not own morphology generation, TXL cache behavior, or evaluation
  scoring beyond the slot fields they consume.
- Reviewer verifies that every downstream subtask references the slot schema and
  that no slot meaning changes across variants.

## Failure Exit

If a variant needs a different action dimension or reuses a slot for a different
joint meaning, stop and route back for scope clarification.

## Log

- 2026-05-29 Opened after the G1-like-only decision.
- 2026-05-29 Implemented local fixed 29-slot G1-like action schema and
  selector/mask mappings in `src/h200_locomotion_lab/robots/g1like_slots.py`.
- 2026-05-29 Added local contract tests in
  `tests/test_task038_g1like_slot_contract.py` covering slot uniqueness,
  29DoF command_mujoco round-trip, 27DoF no-hand masking, missing-slot fill,
  duplicate semantic rejection, and unknown joint rejection.
- 2026-05-29 Verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_g1like_slot_contract.py`
  -> `6 passed in 0.08s`.
- 2026-05-29 Addressed reviewer blocking finding: robot joint orders now reject
  bare semantic names and require canonical non-empty names ending exactly
  `_joint`; added malformed/bare-name tests. Clarified that current 002 scope is
  fixed 29-slot body-only G1/SONIC `command_mujoco`; gripper/hand/finger slots
  require a later explicit action-dim extension.
- 2026-05-29 Verification after reviewer fix:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_g1like_slot_contract.py`
  -> `7 passed in 0.07s`.
- 2026-05-29 Review subagent rechecked the malformed-name and gripper-scope
  fixes and reported no blocking findings.

## Review

Status: local closed with reviewer confirmation for the fixed 29-slot body-only
G1-like action schema. This is not a morphology/H200 load pass; subtask `003`
must still consume this schema.

Evidence:

- Local schema file: `src/h200_locomotion_lab/robots/g1like_slots.py`.
- Local test file: `tests/test_task038_g1like_slot_contract.py`.
- Command result: `7 passed in 0.07s`.
