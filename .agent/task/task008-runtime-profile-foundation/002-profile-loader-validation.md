# Route

Task: task008-runtime-profile-foundation

Goal: load, validate, and compile the YAML profile once at initialization time.

Pass condition:

- `pyyaml` is a core dependency.
- Profile dataclasses validate action dimension and array lengths.
- Mapping arrays are checked as 0..28 permutations.
- Compiled profile exposes hot-path tuples or tensors without dict/name lookup.
- Invalid profile fixtures fail with useful errors.

Fail condition:

- Per-step code reads YAML.
- Per-step code looks up joint names.
- Loader accepts mismatched array lengths or non-permutation mappings.

# Log

- 2026-05-08: Pending YAML profile.
- 2026-05-08: Implemented the 002 minimum closed loop. Added `pyyaml>=6.0`
  to core dependencies and introduced `h200_locomotion_lab.robots` with frozen
  dataclass profile objects plus a YAML/dict loader.
- 2026-05-08: Loader validates the Unitree G1 SONIC profile at initialization
  time: `dof_count == 29`, command/policy joint array lengths, command/policy
  joint set consistency, no hand/finger joints, mapping arrays as `0..28`
  permutations, inverse mapping consistency, mapping-to-joint-name consistency,
  control array lengths, and non-empty `metadata.source`.
- 2026-05-08: Compiled profile stores hot-path fields as tuples:
  command/policy joint orders, policy/command mappings, default angles, action
  scales, kp, kv, and force limits. No scalar/tensor action bridge and no
  runtime migration were implemented in this subtask.
- 2026-05-08: Added `tests/test_robot_profile_loader.py`; the positive test
  loads `configs/robots/unitree_g1_29dof_sonic.yaml` through the loader, and
  invalid fixtures are temporary dict copies created inside the tests.
- 2026-05-08: Verification:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_robot_profile_loader.py tests\test_unitree_g1_29dof_sonic_profile_yaml.py -q -p no:cacheprovider`
  -> `12 passed in 0.36s`.
- 2026-05-08: Related regression:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_sonic_g1_policy_bridge.py tests\test_genesis_adapter.py tests\test_unitree_g1_29dof_sonic_profile_yaml.py tests\test_robot_profile_loader.py -q -p no:cacheprovider`
  -> `29 passed in 0.34s`.
- 2026-05-08: Full local verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider`
  -> `88 passed in 0.55s`.
- 2026-05-08: Lint attempt:
  `$env:PYTHONPATH='src'; python -m ruff check pyproject.toml src\h200_locomotion_lab\robots tests\test_robot_profile_loader.py`
  -> failed because the current interpreter has no `ruff` module installed.
- 2026-05-08: Independent reviewer `Kepler` found no blocking issues. Reviewer
  ran:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_robot_profile_loader.py -q -p no:cacheprovider`
  -> `11 passed in 0.21s`.
- 2026-05-08: Router final verification:
  `$env:PYTHONPATH='src'; python -m pytest tests\test_robot_profile_loader.py tests\test_unitree_g1_29dof_sonic_profile_yaml.py -q -p no:cacheprovider`
  -> `12 passed in 0.30s`.
- 2026-05-08: Router full local verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider`
  -> `88 passed in 0.63s`.

# Review

Status: passed.

Evidence:

- `pyyaml>=6.0` is now a core dependency in `pyproject.toml`.
- `load_robot_profile()` reads the YAML once and returns a
  `CompiledRobotProfile`; no per-step runtime code was migrated.
- Tests cover the real YAML load path plus invalid dict fixtures for action
  dimension, array length, joint set mismatch, mapping permutation failures,
  inverse mapping failure, mapping/name mismatch, missing metadata source, and
  hand/finger joint rejection.
- Independent reviewer accepted the subtask and confirmed there is no 003/004
  action bridge or 005 runtime migration in this change.

Not done:

- No task008/003 scalar action bridge.
- No task008/004 tensor action bridge.
- No task008/005 runtime migration to the loader.
