# 002: No-Update Rollout Probe

## Goal

Add a standalone no-update probe tool that uses the same standing G1 env config
as the PPO smoke path, but does not call `compute_gae` or `ppo_update`.

## Route

1. Reuse the task015/task017 G1 27DoF no-hand standing defaults:
   `tall_crouch`, root z `1.20`, action scale `0.10`, standing-only
   command ranges, and the same vectorized Genesis backend config shape.
2. Add a standalone no-update rollout tool:
   `src/h200_locomotion_lab/tools/g1_no_update_ppo_causality.py`.
3. Add action modes:
   - `zero_action`;
   - `untrained_mean_action`;
   - `untrained_sampled_action`.
4. Run `chunks * chunk_steps` env steps, defaulting to `50 * 32`, with seed
   `0` and no calls to `compute_gae` or `ppo_update`.
5. Write `metrics.jsonl`, `summary.json`, and `config.json` under
   `outputs/task018/no_update_ppo_causality/<run_id>/`.
6. Add focused tests for mode parsing, no-update behavior, summaries, path
   guard, and action stats.

## Log

- Added `g1_no_update_ppo_causality.py`.
- The probe records one JSONL row per chunk with reset/tilt/height counters,
  root height/upright metrics, normalized action stats including top RMS
  joints, throughput, tensor device fields, and run metadata.
- `config.json` and `summary.json` include physical GPU, logical CUDA device,
  `CUDA_VISIBLE_DEVICES`, mode, seed, standing env defaults, and no-update
  marker.
- Added `tests/test_g1_no_update_ppo_causality.py` covering defaults, mode
  parsing, no-update import/call contract, artifact summary generation,
  path guard behavior, and action stat expectations.
- Blocking review fix: removed the probe's dependency on
  `g1_curriculum_ppo_smoke`, moved the needed tiny validators/output helpers,
  CUDA isolation guard, warmup helper, and normalized action stats into the
  probe, and switched reset-pose/backend constants to direct imports.
- Strengthened the probe tests so the probe source must not contain
  `g1_curriculum_ppo_smoke`, `compute_gae`, `ppo_update`, or
  `collect_rollout`, and path-guard tests now patch the probe-local
  `PROJECT_PREFIX`.
- Verification:
  - Initial `python -m pytest tests/test_g1_no_update_ppo_causality.py -q`
    failed because the local environment did not have `src` on
    `PYTHONPATH`.
  - `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_no_update_ppo_causality.py -q`
    passed: `5 passed, 3 skipped`.
  - `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_no_update_ppo_causality.py tests/test_g1_curriculum_ppo_smoke.py -q`
    passed: `15 passed, 7 skipped`.
  - `$env:PYTHONPATH='src'; python -m pytest -q` passed:
    `220 passed, 11 skipped`.
  - Pytest emitted a cache warning because this worktree refused writes to
    `.pytest_cache`; test results still passed.
  - Review fix: `Get-Content src/h200_locomotion_lab/tools/g1_no_update_ppo_causality.py | Select-String -Pattern 'g1_curriculum_ppo_smoke|compute_gae|ppo_update|collect_rollout|curriculum'`
    returned no matches.
  - Review fix: `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_no_update_ppo_causality.py -q`
    passed: `6 passed, 3 skipped`; pytest emitted the same cache warning.
  - Review fix: `$env:PYTHONPATH='src'; python -m pytest -q` passed:
    `220 passed, 11 skipped`; pytest emitted the same cache warning.

## Review

Status: implemented for subtask 002 only.

Notes:

- No changes were made to `GenesisG1SceneBackend`.
- No assets, checkpoints, datasets, or upstream repositories were downloaded.
- The no-update tool builds an untrained actor-critic only for the two
  untrained policy modes; `zero_action` uses direct zero normalized actions.
- Torch-dependent fake-runtime tests are skipped in this local Windows
  environment because `torch` is unavailable, but the full suite passes with
  those skips. The H200 run path still uses `require_torch()`.
