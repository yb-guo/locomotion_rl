# 006: OOD And Failure Grid Eval

## Route

Evaluate the accepted MLP checkpoint beyond the training distribution without
changing the first-pass training objective.

Eval groups:

1. Clean eval.
   - No motor randomization or failure.
   - Confirms baseline walking was not destroyed.

2. Motor-only randomized eval.
   - Uses the task029 training distribution.
   - Confirms in-distribution robustness.

3. Doubled motor randomization holdout.
   - Widens motor ranges relative to training.
   - Mimics LocoFormer-style OOD evaluation without changing topology.

4. Dead-motor grid.
   - Force one leg motor dead at a time.
   - Report per-joint survival, tracking, yaw, gravity, root height, and fall
     metrics.

5. Optional diagnostic holdouts.
   - Locked joint and stuck command are eval-only diagnostics in task029.
   - They are not training randomization for first acceptance.

## Minimal Closed Loop

Feedback loop:

1. Load one saved checkpoint.
2. Run all eval groups with deterministic seeds and fixed commands.
3. Save one JSON per group plus an aggregate summary.
4. Include actual motor scale/failure mask settings in JSON diagnostics.
5. Compare clean, in-distribution, doubled holdout, and grid metrics.

Pass:

- Clean eval remains stable.
- In-distribution motor-randomized eval meets predeclared thresholds.
- Doubled holdout and dead-motor grid produce complete reports even if some
  joints fail.
- Eval summaries include exact fault settings used per run.

Fail:

- Eval requires privileged actor fault inputs.
- Only training reward is reported.
- Failed grid cases are omitted instead of reported.
- The same checkpoint cannot be evaluated across all groups.

Evidence:

- Planned output:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task029/ood_failure_grid_eval/`.

## Log

- 2026-05-19 Opened to separate convergence claims from robustness claims.
  The first task029 pass can succeed even if some grid joints remain hard
  failures, as long as the grid is complete and diagnostic.

## Review

Status: pending.

This subtask should produce the table that tells us whether task029 is enough
for the current MLP or whether the next task needs long-context adaptation.
Do not hide failure cases; the grid is useful precisely because it names which
motors break the gait.
