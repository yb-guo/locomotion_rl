# Subtask 001: Bundle Extractor And Alignment Report

## Route

- Add a standalone tool under `src/h200_locomotion_lab/tools/`.
- Do not import Genesis in the tool.
- Load existing robot profile YAML through current structured loaders where
  possible.
- Parse MJCF/XML with the standard XML parser when an asset path exists.
- Emit stable JSON containing:
  - `sonic_29dof_profile`;
  - `genesis_27dof_training_profile`;
  - `mapped_control_comparison`;
  - `root_pose_defaults`;
  - `contact_friction_solver`;
  - `missing`;
  - `alignment_status`.
- Add focused tests that verify:
  - mapped 29DoF SONIC control arrays match the 27DoF profile after removing
    `waist_roll_joint` and `waist_pitch_joint`;
  - backend runtime defaults are represented;
  - absent XML/contact values are reported as missing, not silently passed.

## Log

- 2026-05-12 Ready for coding subagent implementation.
- 2026-05-12 Coding subagent implemented
  `h200_locomotion_lab.tools.g1_genesis_alignment_bundle` and focused tests.
  Router fix preserved the default remote `/root/...` asset path in local JSON
  output and added explicit missing records for unrepresented training
  profile/backend contact-friction-solver config.
- 2026-05-12 Local verification:
  `PYTHONPATH=src python -m pytest tests/test_g1_genesis_alignment_bundle.py
  tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q
  -p no:cacheprovider` -> 33 passed.
- 2026-05-12 Local JSON evidence written to
  `outputs/task021/genesis_alignment_bundle/local_profile_report.json`.
  Summary: `mapped_control_match=true`, `xml_asset_present=false`,
  `missing_count=7`.
- 2026-05-12 Read-only reviewer found a blocking issue: timing was serialized
  but not explicitly compared or marked missing for SONIC/MJCF sources. Fixed
  by adding `control_timing_comparison` and tests for Genesis timing,
  missing SONIC profile timing, and missing MJCF decimation/policy-rate
  semantics. Re-run:
  `PYTHONPATH=src python -m pytest tests/test_g1_genesis_alignment_bundle.py
  tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q
  -p no:cacheprovider` -> 34 passed. Import boundary check:
  `import h200_locomotion_lab.tools.g1_genesis_alignment_bundle` leaves
  `genesis` and `torch` absent from `sys.modules`. Updated local report:
  `mapped_control_match=true`, `xml_asset_present=false`, `missing_count=13`;
  Genesis timing is self-consistent at `0.005 * 4 -> 50 Hz`.

## Review

Status: passed after re-review.

- Blocking finding fixed: explicit timing comparison and missing records were
  added for SONIC/MJCF timing semantics.
- Re-review found no blocking issues.
- Residual risks:
  - future non-integer timing drift could be hidden by rounded consistency
    check;
  - malformed MJCF `option.timestep` would currently parse as missing rather
    than `invalid_value`;
  - contact/friction/solver values remain unproven until the prepared H200
    asset is available to the report.
