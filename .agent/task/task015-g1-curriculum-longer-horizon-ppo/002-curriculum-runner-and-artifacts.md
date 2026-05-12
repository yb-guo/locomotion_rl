# 002: Curriculum Runner And Artifacts

## Goal

Add an independent G1 curriculum PPO runner that reuses task014 PPO core and
env semantics.

## Route

1. Implement a new tool under `src/h200_locomotion_lab/tools`.
2. Use a single policy/optimizer per seed across ordered curriculum stages.
3. Keep stages explicit and small:
   - `standing`;
   - `small_vx`;
   - `small_yaw`;
   - `small_vxyaw`.
4. Write `config.json`, `metrics.jsonl`, `summary.json`, and
   `final_checkpoint.pt`.
5. Record per-update metrics with stage and local/global update indexes.
6. Stop later stages when upstream stage criteria fail.

## Log

- 2026-05-09 Coding subagent implemented
  `src/h200_locomotion_lab/tools/g1_curriculum_ppo_smoke.py`.
- 2026-05-09 Runner keeps one actor-critic policy and one Adam optimizer per
  seed across explicit ordered stages: `standing`, `small_vx`, `small_yaw`,
  `small_vxyaw`.
- 2026-05-09 Runner writes `config.json`, `metrics.jsonl`, `summary.json`, and
  `final_checkpoint.pt` under `PROJECT_PREFIX`-guarded run directories.
- 2026-05-09 Added focused local tests in
  `tests/test_g1_curriculum_ppo_smoke.py` for CLI defaults, stage order,
  output path guard, artifact/metrics schema, and standing-failure skip rules.
- 2026-05-09 Verification:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_curriculum_ppo_smoke.py tests/test_g1_ppo_smoke.py tests/test_ppo_loop.py -q -p no:cacheprovider`
  passed with `14 passed, 6 skipped`. Skips are local no-torch paths.

## Review

Status: passed.

- Local focused tests passed.
- Router local full pytest passed: `196 passed, 6 skipped`.
- Read-only reviewer subagent found no blocking issues:
  - single policy/optimizer per seed across stages;
  - upstream stage failure skips later stages;
  - output path guard is present;
  - required artifacts are written;
  - required per-update metrics are present;
  - boundary scope is clean.
- H200 focused tests, H200 dev probe, and H200 final smoke remain pending.
