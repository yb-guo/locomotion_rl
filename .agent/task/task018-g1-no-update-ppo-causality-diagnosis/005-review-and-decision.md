# 005: Review And Decision

## Goal

Review evidence and choose the next engineering direction.

## Route

1. Read-only reviewer checks:
   - boundary compliance;
   - probe is truly no-update;
   - env/reset semantics match task017 standing config;
   - H200 evidence;
   - stop-rule application.
2. Fix blocking findings through coding subagent.
3. Record final causal decision.

## Log

- 2026-05-11 Coding subagent implemented the standalone no-update probe.
- 2026-05-11 Read-only reviewer found one blocking boundary issue: the probe
  imported `g1_curriculum_ppo_smoke`, which transitively imported PPO update
  helpers.
- 2026-05-11 Coding subagent fixed the boundary by removing the curriculum
  import and moving tiny validators/output helpers into the standalone probe.
- 2026-05-11 Read-only reviewer found no blocking findings after the fix.
- 2026-05-11 Local full pytest passed: `220 passed, 11 skipped`.
- 2026-05-11 H200 focused tests passed: `43 passed in 5.06s`.
- 2026-05-11 H200 no-update matrix completed all three modes.
- 2026-05-11 Final read-only reviewer found no blocking findings and said the
  task can be marked passed.

Decision:

Task018 supports **control/reset semantics instability before PPO updates**.

Evidence:

- `zero_action` no-update reproduced the reset wave:
  `first_tilt_chunk=2`, `max_reset_count=1024`,
  `mean_reset_count=348.16`, `final_reset_count=1024`,
  `final_tilt_bad_count=1024`.
- `untrained_mean_action` reproduced the same reset wave even with tiny
  normalized actions: final action abs mean/max `0.0024/0.0129`.
- `untrained_sampled_action` also reproduced the same reset wave with action
  abs mean/max `0.0652/0.3994`.
- All three modes had `final_termination_height_bad_count=0`, so this remains
  a tilt/fall reset path, not hard height reset.
- The no-update reset profile matches task017 PPO u50 targeted evidence:
  `first_tilt=2`, `mean_reset=348.16`, `final_reset=1024`, and
  `final_tilt=1024`.

Stop-rule outcome:

- PPO-update isolation was skipped because `zero_action` no-update already
  falls.
- The next task should target reset/default-pose/PD/control semantics, not
  PPO reward/value/advantage first.

## Review

Status: passed.

- Boundary compliance passed.
- No accidental PPO update or `collect_rollout` path was found in the probe.
- No `GenesisG1SceneBackend` changes were made.
- Stop-rule application passed.
- Residual risk: the reviewer relied on recorded H200 verification instead of
  rerunning it during review.
