# 002: Standing-Only Runner And Action Stats

## Goal

Add the tooling needed to run task017 standing-only action/control semantics
diagnostics.

## Route

1. Preserve task015/task016 defaults.
2. Add a way to run only the `standing` curriculum stage.
3. Add action statistics to metric rows and stage summaries.
4. Add a task017 ablation runner with one subprocess per variant.
5. Add focused tests for stage selection, action stats, variants, and stop
   behavior.

## Log

- 2026-05-11 Implemented `--stage-names` in
  `g1_curriculum_ppo_smoke.py`. Empty default preserves the full task015/task016
  curriculum; `--stage-names standing` runs only standing.
- 2026-05-11 Added normalized action stats to metric rows and stage summaries:
  `action_abs_mean`, `action_abs_max`, `action_std`, and top joint RMS entries
  keyed by the G1 profile `actuator_order`.
- 2026-05-11 Added `h200_locomotion_lab.tools.g1_action_control_semantics`.
  It defaults to `outputs/task017/action_control_semantics`, runs one
  curriculum subprocess per variant, defaults variants to standing-only, and
  stops before variants if baseline does not reproduce tilt/reset waves.
- 2026-05-11 Router corrected the task017 runner default to
  `updates_per_stage=10` to match the fast feedback-loop contract.
- 2026-05-11 Added focused tests for default preservation, stage selection,
  action stats, task017 variants, stop rule, and subprocess success/failure
  summary handling.
- 2026-05-11 Verification:
  - `PYTHONPATH=src python -m pytest -o cache_dir=.test_tmp_pytest_cache tests/test_g1_curriculum_ppo_smoke.py tests/test_g1_action_control_semantics.py`
    -> `17 passed, 4 skipped`.
  - `PYTHONPATH=src python -m pytest -p no:cacheprovider`
    -> `214 passed, 8 skipped`.
- 2026-05-11 Router verification:
  - `PYTHONPATH=src python -m pytest -o cache_dir=.test_tmp_pytest_cache tests/test_g1_curriculum_ppo_smoke.py tests/test_g1_action_control_semantics.py`
    -> `17 passed, 4 skipped`.
  - `PYTHONPATH=src python -m pytest -q -p no:cacheprovider`
    -> `214 passed, 8 skipped`.
- 2026-05-11 Read-only reviewer found no blocking findings. Residual risk:
  action RMS stats are normalized policy actions, not backend-masked/scaled
  applied actions.

## Review

Status: passed.

- Full task017 is not marked passed here; H200 evidence is recorded in later
  subtasks.
