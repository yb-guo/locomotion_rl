# Route

Task: task008-runtime-profile-foundation

Goal: implement batched action bridge math for training/evaluation hot paths.

Pass condition:

- Batch raw action shape `[N, 29]` maps to target shape `[N, 29]`.
- Batch size 1 output matches scalar bridge output.
- Mapping uses precompiled indices.
- No per-frame dataclass allocation is required in the tensor hot path.
- Implementation is compatible with the project's current lightweight
  dependency policy; optional torch path may be guarded if torch is unavailable.

Fail condition:

- Tensor path loops through joint names per step.
- Tensor path requires real Genesis or hardware.
- Tensor path diverges from scalar action semantics.

# Log

- 2026-05-08: Pending scalar bridge.

# Review

Status: pending.

