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
- 2026-05-08: Implemented `NumpyTensorActionBridge` under runtime with
  initialization-time NumPy array compilation for command-to-policy indices,
  default command angles, and action scales. Hot path validates rank, `[N, 29]`
  width, and finite values, then computes batched command targets without
  per-frame dataclass allocation or joint-name lookup.
- 2026-05-08: Verification passed:
  `PYTHONPATH=src python -m pytest -q -p no:cacheprovider
  tests\test_numpy_tensor_action_bridge.py tests\test_scalar_action_bridge.py
  tests\test_robot_profile_loader.py` -> `30 passed in 1.28s`.
- 2026-05-08: Full suite passed:
  `PYTHONPATH=src python -m pytest -q -p no:cacheprovider` ->
  `107 passed in 1.08s`.

# Review

Status: passed.

- 2026-05-08: Read-only reviewer reported no blocking findings and no
  suggestions. Reviewer verified scoped changes, no SONIC bridge migration, and
  `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_numpy_tensor_action_bridge.py tests/test_scalar_action_bridge.py`:
  `19 passed in 0.56s`.

