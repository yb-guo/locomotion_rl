# 002: Ablation Runner And Summary

## Goal

Add instrumentation and an ablation runner/mode that compares seed-0 variants
against the task015 baseline.

## Route

1. Add per-stage summary fields to expose reset waves.
2. Add a small ablation CLI or mode under `h200_locomotion_lab.tools`.
3. Keep variants one-variable-at-a-time.
4. Write artifacts under `outputs/task016/tilt_reset_ablation/<run_id>/`.
5. Preserve task015 runner behavior for existing defaults.

## Log

- 2026-05-09 Coding subagent added per-stage reset-wave diagnostics to
  `g1_curriculum_ppo_smoke.py` without changing task015 CLI defaults.
- 2026-05-09 Added `h200_locomotion_lab.tools.g1_tilt_reset_ablation` with
  seed-0 default, output root `outputs/task016/tilt_reset_ablation`, and
  one-variable variants: `baseline`, `lr_1e4`,
  `termination_penalty_neg5`, `action_rate_penalty_high`.
- 2026-05-09 Added focused local tests for diagnostic summaries, variant
  construction, aggregate artifacts, and task015 default preservation.
- 2026-05-09 Read-only reviewer found two blocking orchestration issues:
  baseline mismatch did not stop later variants, and variant completion did not
  require `run_smoke` pass evidence.
- 2026-05-09 Router fixed both blocking issues: baseline mismatch now stops the
  matrix, failed `run_smoke` summaries mark the variant failed, and root
  height/upright diagnostics are included in the aggregate summary.
- 2026-05-09 H200 v1 exposed an additional runner issue:
  `GenesisException:Genesis already initialized.` after the baseline variant.
  Router changed the ablation runner so each variant runs the task015 runner in
  an isolated subprocess and writes per-variant stdout/stderr under
  `outputs/task016/tilt_reset_ablation/<run_id>/logs/`.
- 2026-05-09 Verification:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_tilt_reset_ablation.py tests/test_g1_curriculum_ppo_smoke.py tests/test_g1_ppo_smoke.py tests/test_ppo_loop.py -q -p no:cacheprovider`
  passed with 21 passed, 6 skipped.
- 2026-05-09 Verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider`
  passed with 203 passed, 6 skipped.

## Review

Status: passed after blocking review fixes.

- Implementation stays inside the allowed files and does not modify
  `GenesisG1SceneBackend`.
- The ablation runner reuses the task015 runner through subprocess calls; it
  does not duplicate the PPO loop and avoids Genesis singleton reuse.
- No downloads, rendering, upstream repos, datasets, checkpoints, or
  `/mnt/workspace*` writes were used.
