# Task 016: G1 Long-Horizon Tilt Reset Ablation

## Goal

Diagnose recurring long-horizon `tilt_bad` resets observed in task015 final
curriculum PPO smoke.

Task015 proved the curriculum runner works, but final H200 rows still showed
reset waves:

- `reset_count=1024`;
- `tilt_bad_count=1024`;
- `termination_height_bad_count=0`.

This task should identify the most likely cause class before any broader reward
redesign. It is not a walking-quality claim.

## Scope

- Branch: `codex/task016-g1-long-horizon-tilt-reset-ablation`.
- Worktree:
  `../_worktrees/h200-locomotion-lab-task016-g1-long-horizon-tilt-reset-ablation`.
- Base: stacked on `codex/task015-g1-curriculum-longer-horizon-ppo` commit
  `b2c91ca`.
- Reuse task015 `g1_curriculum_ppo_smoke.py`.
- Add focused diagnostics and a small ablation runner or mode.
- Compare a small H200 matrix against the task015 baseline.
- Keep the first ablation cheap enough to run:
  - seed `0`;
  - `updates_per_stage=50`;
  - variants:
    - baseline;
    - lower LR (`1e-4`);
    - stronger termination penalty (`-5`);
    - stronger action-rate penalty.

## Non-Goals

- No walking-quality claim.
- No full reward redesign.
- No LocoFormer integration.
- No SONIC integration.
- No ONNX export.
- No rendering/GIF/video.
- No domain randomization.
- No dataset/checkpoint/asset/upstream download.
- No writes/deletes under `/mnt/workspace` or `/mnt/workspace1`.
- No change to `GenesisG1SceneBackend`.

## H200 Protocol

Remote commands must use:

```bash
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'
```

All remote code, outputs, and intermediate files must stay under:

```text
/root/agent_workspace/project
```

Default GPU:

```text
CUDA_VISIBLE_DEVICES=1
physical_gpu=1
logical_cuda_device=cuda:0
```

Run dir:

```text
outputs/task016/tilt_reset_ablation/<run_id>/
```

## Diagnose Loop

### Feedback Loop

Use task015 H200 final as the original repro. Add a runner summary that can
show, per seed/stage:

- first update with `tilt_bad_count > 0`;
- max/mean/final `reset_count`;
- max/final `approx_kl`;
- final entropy;
- final root height and upright metrics;
- throughput.

### Ranked Hypotheses

1. **PPO schedule drift**
   - Prediction: lower LR or smaller clip reduces first tilt reset update,
     max reset count, or final reset count.
2. **Tilt/reset reward is too weak**
   - Prediction: stronger termination penalty or upright reward reduces reset
     waves without changing simulator semantics.
3. **Action smoothness is too weak**
   - Prediction: stronger action-rate penalty reduces reset waves and possibly
     KL/entropy.
4. **Curriculum command range is too coarse**
   - Prediction: smaller command ranges reduce reset waves after stage
     transition. This is lower priority because task015 final rows also fail in
     `standing`.

## Stop Rules

- If the baseline no longer reproduces tilt reset waves, stop and record the
  mismatch before tuning.
- Change one variable per variant.
- Do not run final three-seed experiments until a seed-0 ablation points to a
  cause class.
- Do not mark passed without local tests, H200 focused tests, H200 ablation
  evidence, and read-only review.

## Acceptance

- Router creates task/subtask docs before coding.
- Coding subagent implements instrumentation/ablation runner.
- Read-only reviewer subagent reviews boundary, correctness, and evidence.
- Local focused tests pass.
- Local full pytest passes.
- H200 focused tests pass.
- H200 seed-0 baseline reproduces recurring tilt reset waves, or mismatch is
  recorded and the task stops.
- H200 seed-0 ablation compares at least three variants against baseline.
- Decision states which hypothesis is most supported, or why evidence is
  inconclusive.

# Route

1. `001-diagnosis-contract.md`
2. `002-ablation-runner-and-summary.md`
3. `003-h200-baseline-reproduction.md`
4. `004-h200-ablation-matrix.md`
5. `005-review-and-decision.md`

# Log

- 2026-05-09 Created task016 branch/worktree from task015 commit `b2c91ca`.
- 2026-05-09 Router created diagnosis contract before coding.
- 2026-05-09 Coding subagent added reset-wave diagnostics and an ablation
  runner.
- 2026-05-09 Read-only reviewer found blocking runner orchestration issues;
  Router fixed baseline stop behavior and failed-summary handling.
- 2026-05-09 Local focused tests passed with 21 passed, 6 skipped.
- 2026-05-09 Local full pytest passed with 203 passed, 6 skipped.
- 2026-05-09 H200 focused tests passed with 27 passed in 23.01s.
- 2026-05-09 H200 v1 reproduced baseline but found same-process Genesis
  reinitialization failure for later variants.
- 2026-05-09 Router changed variant execution to one subprocess per variant.
- 2026-05-09 H200 v2 completed all variants:
  `/root/agent_workspace/project/h200-locomotion-lab-task016-g1-long-horizon-tilt-reset-ablation/outputs/task016/tilt_reset_ablation/h200-gpu1-seed0-updates50-v2`.
- 2026-05-09 Final read-only reviewer found no blocking findings.

# Review

Status: passed.

- Baseline reproduced: all stages reached `final_reset_count=1024` and
  `final_tilt_bad_count=1024`, with `termination_height_bad_count=0`.
- All three tested variants kept recurring tilt reset waves.
- Decision: evidence does not support LR, termination penalty, or action-rate
  penalty as sufficient fixes; next diagnosis should target action
  amplitude/control semantics and upright/joint-deviation shaping.
