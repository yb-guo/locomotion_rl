# Route

Task: task008-runtime-profile-foundation

Goal: add the authoritative YAML robot/control profile for SONIC G1 29DoF.

Pass condition:

- `configs/robots/unitree_g1_29dof_sonic.yaml` exists.
- It contains 29 command joint names and 29 policy joint names.
- It contains policy/command mapping arrays.
- It contains SONIC G1 default angles, action scales, kp, kv, and force limits
  in command order.
- The file records that the constants mirror official SONIC deploy behavior.

Fail condition:

- Runtime constants are hand-copied into multiple sources of truth.
- YAML lacks enough information to compile the action bridge.
- Dexterous hand joints are mixed into the 29DoF body profile.

# Log

- 2026-05-08: Opened route from architecture discussion.
- 2026-05-08: Added `configs/robots/unitree_g1_29dof_sonic.yaml` as the
  authoritative SONIC G1 29DoF body-only profile for task008/001. The profile
  records command/MuJoCo joint order, raw policy/IsaacLab joint order,
  bidirectional index mappings, default angles, action scales, kp, kv, and
  force limits. Constants mirror the existing runtime sources in
  `src/h200_locomotion_lab/sonic/g1_policy_bridge.py`,
  `src/h200_locomotion_lab/envs/genesis_adapter.py`, and
  `src/h200_locomotion_lab/tools/sonic_reference_replay_smoke.py`.
- 2026-05-08: Added
  `tests/test_unitree_g1_29dof_sonic_profile_yaml.py`; the test reads the YAML
  file directly with `pytest.importorskip("yaml")` and checks 29 DoF shape,
  command/policy joint consistency, mapping permutations, control array lengths,
  no hand/finger joints, and official SONIC deploy mirror metadata.
- 2026-05-08: Verification:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_unitree_g1_29dof_sonic_profile_yaml.py -q -p no:cacheprovider`
  -> `1 passed in 0.11s`.
- 2026-05-08: Verification:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_sonic_g1_policy_bridge.py tests\test_genesis_adapter.py tests\test_unitree_g1_29dof_sonic_profile_yaml.py -q -p no:cacheprovider`
  -> `18 passed in 0.14s`.
- 2026-05-08: A direct `python -m pytest
  tests\test_unitree_g1_29dof_sonic_profile_yaml.py -q` run failed during
  collection because this shell did not have the package installed or
  `PYTHONPATH=src`; pytest cache writes also hit local workspace permission
  warnings. The verified commands above use `PYTHONPATH=src` and disable the
  cache provider.
- 2026-05-08: Reviewer follow-up closed: strengthened
  `tests/test_unitree_g1_29dof_sonic_profile_yaml.py` so YAML `kp`, `kv`, and
  `force_limits` values are compared with `SONIC_G1_KPS`, `SONIC_G1_KDS`, and
  `SONIC_G1_FORCE_LIMITS` from
  `src/h200_locomotion_lab/tools/sonic_reference_replay_smoke.py`, the same
  values consumed by `apply_sonic_g1_motor_config`. Existing checks against
  `g1_policy_bridge.py` for default angles, action scales, and mappings remain.
- 2026-05-08: Verification after reviewer follow-up:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_unitree_g1_29dof_sonic_profile_yaml.py -q -p no:cacheprovider`
  -> `1 passed in 0.09s`.
- 2026-05-08: Related regression after reviewer follow-up:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_sonic_g1_policy_bridge.py tests\test_genesis_adapter.py tests\test_unitree_g1_29dof_sonic_profile_yaml.py -q -p no:cacheprovider`
  -> `18 passed in 0.13s`.

# Review

Status: passed.

Evidence:

- YAML exists at `configs/robots/unitree_g1_29dof_sonic.yaml`.
- The profile keeps command arrays in MuJoCo/hardware order and raw policy
  action arrays in policy/IsaacLab order.
- Test coverage reads the YAML file itself and compares shape/order/mapping
  against current source constants without adding a loader or schema
  implementation.
- Test coverage compares YAML default angles, action scales, and policy/command
  mappings against `g1_policy_bridge.py`; it also compares YAML kp, kv, and
  force limits against the current SONIC G1 motor config constants used by
  `apply_sonic_g1_motor_config`.

Not done:

- No 002 loader/runtime migration was implemented.
- No dependency was added for PyYAML; the YAML test uses
  `pytest.importorskip("yaml")`. This is an explicit 001 test limitation only,
  not evidence that runtime YAML loading is complete.
