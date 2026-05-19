# Subtask 004: Backend Contact Solver Boundary

## Route

- Add a local, optional Genesis rigid contact/solver config boundary to
  `VectorizedGenesisConfig`.
- Keep default backend construction unchanged: no `rigid_options` passed unless
  the user supplies explicit fields.
- Expose the requested/default backend contact solver boundary in the alignment
  JSON so it no longer reports the backend as a blanket missing config surface.
- Verify with fake Genesis first, then with a guarded H200 real-Genesis enum
  probe. Do not run PPO, rendering, downloads, or change
  `GenesisG1SceneBackend`.

## Log

- 2026-05-12 Added `GenesisRigidContactSolverConfig` with optional primitive
  fields matching the H200 Genesis `gs.options.RigidOptions` introspection.
- 2026-05-12 Added `_make_rigid_options()` in `VectorizedGenesisBackend`; it
  calls `gs.options.RigidOptions(**kwargs)` only when at least one field is set
  and the class exists, then passes `rigid_options` to `gs.Scene`.
- 2026-05-12 Added backend/config report data for configured, requested,
  applied, unset, and missing contact/solver config state.
- 2026-05-12 Extended `g1_genesis_alignment_bundle.py` to include
  `vectorized_genesis_backend_contact_solver_config` and replace the old
  `vectorized_genesis_backend.contact_friction_solver_config` blanket missing
  entry with an explicit unset-default report.
- 2026-05-12 Focused local command:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_vectorized_genesis_backend.py tests/test_g1_genesis_alignment_bundle.py -q -p no:cacheprovider`
  -> 22 passed.
- 2026-05-12 Router reran expanded local related tests:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_vectorized_genesis_backend.py tests/test_g1_genesis_alignment_bundle.py tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q -p no:cacheprovider`
  -> 49 passed.
- 2026-05-12 H200 real Genesis found that `gs.options.RigidOptions` rejects
  raw `constraint_solver` strings/ints and expects enum objects such as
  `gs.constraint_solver.Newton` or `gs.constraint_solver.CG`. Fixed the local
  backend boundary to map configured strings (`Newton`/`newton`, `CG`/`cg`) to
  Genesis enum values before constructing `RigidOptions`, while keeping the
  requested string in config/report output.
- 2026-05-12 Router reran expanded local related tests after the enum fix:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_vectorized_genesis_backend.py tests/test_g1_genesis_alignment_bundle.py tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q -p no:cacheprovider`
  -> 51 passed.
- 2026-05-12 H200 guarded tests passed:
  `PYTHONPATH=src python -m pytest tests/test_vectorized_genesis_backend.py tests/test_g1_genesis_alignment_bundle.py tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q -p no:cacheprovider`
  -> 51 passed.
- 2026-05-12 H200 real Genesis enum probe used
  `CUDA_VISIBLE_DEVICES=1`, `physical_gpu=1`, `logical_cuda_device=cuda:0` and
  `constraint_solver="Newton"`. Result:
  `applied_rigid_options=True`, `rigid_options_class_present=True`,
  `missing=[]`, `SCENE_BUILT 1 27 27`.
- 2026-05-12 Regenerated H200 alignment report:
  `/root/agent_workspace/project/h200-locomotion-lab-task021-genesis-alignment-bundle/outputs/task021/genesis_alignment_bundle/h200_asset_report.json`.
  Summary: `status=pass`, `mapped_control_match=true`,
  `xml_asset_present=true`, `missing_count=11`, backend rigid-contact config
  remains explicitly `configured=false`/`unset_defaults` by default.

## Review

Status: passed. Read-only review found no blocking findings.

Notes:

- Default behavior remains conservative: no rigid options are created or passed
  when all fields are unset.
- Configured fake-Genesis test verifies supported fields are forwarded to
  `gs.options.RigidOptions` and then to `gs.Scene(rigid_options=...)`.
- Constraint solver strings are now resolved locally before the Genesis API
  call; unsupported strings raise a clear `ValueError`. H200 real Genesis
  accepted the mapped `Newton` enum path and built the scene.
- Residual risk: contact/friction/solver semantics are configurable but still
  unset by default; the task report intentionally keeps this as
  `unset_defaults` evidence rather than claiming solver alignment.
