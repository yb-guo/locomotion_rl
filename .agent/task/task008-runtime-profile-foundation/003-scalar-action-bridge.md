# Route

Task: task008-runtime-profile-foundation

Goal: implement the scalar action bridge from loaded profile data.

Pass condition:

- Raw 29D policy/IsaacLab action maps to 29D command/MuJoCo motor targets.
- Formula matches official SONIC deploy semantics:

```text
target_command[i] =
  default_angles_command[i]
  + raw_action_policy[command_to_policy[i]] * action_scale_command[i]
```

- Outputs match the current `g1_policy_bridge.py` implementation before
  migration.

Fail condition:

- The scalar bridge reads global SONIC constants instead of the loaded profile.
- The bridge clips raw actions unless explicitly configured as a separate safety
  layer.

# Log

- 2026-05-08: Pending profile loader.
- 2026-05-08: Implemented `ScalarActionBridge.from_profile()` and
  `policy_action_to_command_targets()` using compiled profile mapping,
  default angles, and action scales. Verified targeted tests with
  `PYTHONPATH=src python -m pytest -q tests/test_scalar_action_bridge.py tests/test_robot_profile_loader.py tests/test_unitree_g1_29dof_sonic_profile_yaml.py tests/test_sonic_g1_policy_bridge.py`:
  `26 passed, 1 warning in 0.44s` (warning was pytest cache write permission).
  Verified full local suite with
  `PYTHONPATH=src python -m pytest -q -p no:cacheprovider`: `98 passed in 0.65s`.

# Review

Status: passed.

- 2026-05-08: Read-only reviewer reported no blocking findings. Reviewer ran
  `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_scalar_action_bridge.py -p no:cacheprovider`:
  `10 passed in 0.17s`.
  Residual suggestion: add constructor-level invariant checks later if direct
  construction becomes public API beyond `from_profile()`.

