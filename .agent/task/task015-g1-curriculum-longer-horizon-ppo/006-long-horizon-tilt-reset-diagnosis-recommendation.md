# 006: Long-Horizon Tilt Reset Diagnosis Recommendation

## Goal

Recommend the next closed-loop work after task015's longer-horizon curriculum
smoke.

Task015 passed as a runner/curriculum smoke, but final H200 rows still show
recurring tilt resets:

- `reset_count=1024`;
- `tilt_bad_count=1024`;
- `termination_height_bad_count=0`.

This subtask does not implement code. It records the next diagnosis target so
the next coding subtask/task does not drift into broad reward tuning or walking
quality claims.

## Route

1. Treat task015 final evidence as the reproduction.
2. Rank likely causes for recurring long-horizon tilt resets.
3. Recommend the smallest next diagnostic implementation.
4. Define stop rules and acceptance for the next coding unit.

## Ranked Hypotheses

1. **Policy drift / update schedule issue**
   - Evidence: task014 5-update smoke and task015 20-update dev probe can end
     with zero final resets, while task015 50-update final rows show reset
     waves again.
   - Prediction: lower LR, smaller clip, or per-stage reset/restart of optimizer
     reduces final reset waves without changing simulator semantics.

2. **Reward does not price tilt failure strongly enough**
   - Evidence: `termination_height_bad_count=0`, so the failure is upright,
     not height. Current reward can pass smoke while allowing recurring tilt
     reset waves.
   - Prediction: stronger `termination_penalty`, higher upright reward weight,
     or action-rate/joint-deviation penalties reduce `tilt_bad_count`.

3. **Curriculum stage targets are still too coarse for 50 updates**
   - Evidence: final reset waves appear in all stages, including `standing`,
     so this is not only `small_vxyaw`, but velocity/yaw stages may amplify the
     same instability.
   - Prediction: smaller command ranges or longer standing pretrain reduces
     downstream tilt waves.

4. **Diagnostics are not yet sharp enough**
   - Evidence: task015 records reset/height/tilt and PPO metrics, but not
     action norm, action rate, log-std trend, joint deviation distribution, or
     first reset update per stage in summary.
   - Prediction: adding stage summaries will identify whether resets correlate
     with policy std, action jumps, joint drift, or stage transition.

## Recommended Next Unit

Create a new coding subtask or task:

```text
task016-g1-long-horizon-tilt-reset-ablation
```

If staying inside task015, use:

```text
007-tilt-reset-ablation-runner.md
```

Recommended implementation:

1. Add diagnostic summary fields to `g1_curriculum_ppo_smoke.py`:
   - first update where `tilt_bad_count > 0` per seed/stage;
   - max and mean `reset_count` per seed/stage;
   - final and max `approx_kl`;
   - final policy entropy / log-std proxy;
   - action norm and action-rate mean if cheap to collect;
   - joint deviation mean/max if already available in env components.
2. Add CLI knobs only where existing env/PPO semantics already support them:
   - `--upright-reward-scale`;
   - `--termination-penalty`;
   - `--action-rate-penalty-scale`;
   - `--joint-deviation-penalty-scale`;
   - `--lr`;
   - `--clip`.
3. Run a small H200 ablation matrix before any long final:
   - baseline task015 config;
   - lower LR (`1e-4`);
   - stronger termination penalty (`-5`);
   - stronger action-rate penalty;
   - smaller command range if needed.
4. Stop after identifying the best next direction; do not claim walking.

## Stop Rules For Next Unit

- Do not change `GenesisG1SceneBackend`.
- Do not download assets, checkpoints, datasets, or upstream repos.
- Do not use render/GIF/video, SONIC, ONNX, planner, or LocoFormer.
- Do not write/delete under `/mnt/workspace` or `/mnt/workspace1`.
- Do not expand to full reward redesign before the ablation identifies a cause.
- If baseline reproduction does not show tilt resets, stop and record the
  mismatch before tuning.

## Suggested Acceptance For Next Unit

- Local focused tests cover new CLI knobs and summary fields.
- H200 focused tests pass.
- H200 baseline reproduces recurring tilt reset waves or records why it does
  not.
- H200 ablation evidence compares at least three variants against baseline.
- Decision states one of:
  - PPO schedule is the likely cause;
  - reward/penalty weighting is the likely cause;
  - curriculum stage range is the likely cause;
  - evidence is inconclusive and needs sharper instrumentation.

## Log

- 2026-05-09 Created from task015 final evidence after H200 three-seed
  curriculum smoke.
- 2026-05-09 Recommendation: next work should diagnose and reduce long-horizon
  tilt resets, not increase curriculum length.

## Review

Status: passed as recommendation.

- The recommendation is scoped to diagnosis/ablation.
- It preserves task015 boundaries and avoids walking-quality claims.
