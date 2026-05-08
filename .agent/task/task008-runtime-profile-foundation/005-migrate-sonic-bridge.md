# Route

Task: task008-runtime-profile-foundation

Goal: migrate existing SONIC G1 bridge users to profile-backed action/control
data without breaking task006/task007.

Pass condition:

- Existing `g1_policy_bridge.py` either delegates to the loaded profile path or
  is explicitly retained only as compatibility/test reference.
- Genesis SONIC motor config uses the loaded control profile.
- Task007 dry-run uses the loaded control profile.
- Existing tests are updated to compare against profile-backed behavior.

Fail condition:

- Two runtime authorities for default angles/action scales remain active.
- Migration changes task006 action semantics without evidence.

# Log

- 2026-05-08: Pending scalar/tensor bridge validation.
- 2026-05-08: Migrated SONIC G1 action/control runtime users to the
  profile-backed `ScalarActionBridge` without changing task006/task007 action
  semantics. `g1_policy_bridge.py` now loads the default profile once at module
  initialization, exposes compatibility constants from that compiled profile,
  and delegates `sonic_policy_action_to_mujoco_targets()` to the cached scalar
  bridge. `GenesisG1SceneBackend` now creates the same bridge once for
  `sonic_policy_raw`, derives default motor positions from profile control
  defaults, and maps raw actions through the bridge. `G1MotorCommand` now uses
  the cached bridge by default and accepts an explicit bridge for runtime/test
  injection.
- 2026-05-08: Verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests/test_sonic_g1_policy_bridge.py tests/test_scalar_action_bridge.py tests/test_numpy_tensor_action_bridge.py tests/test_genesis_adapter.py tests/test_robot_backend.py tests/test_sonic_g1_deployment_dry_run.py tests/test_robot_profile_loader.py tests/test_unitree_g1_29dof_sonic_profile_yaml.py`
  passed with `57 passed in 0.89s`.
- 2026-05-08: Full local verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider` passed with
  `109 passed in 0.99s`.

# Review

Status: passed.

- 2026-05-08: Read-only reviewer reported no blocking findings and no
  suggestions. Reviewer verified scoped files and ran targeted profile/action
  bridge, Genesis, robot backend, and dry-run tests:
  `57 passed in 1.10s`. Reviewer also ran full local suite:
  `109 passed in 1.19s`.

