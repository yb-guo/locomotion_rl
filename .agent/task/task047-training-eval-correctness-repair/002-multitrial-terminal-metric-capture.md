# 002: Multi-Trial Terminal Metric Capture

## Route

Make Task037 multi-trial metrics read the state that caused a fall/timeout,
not the state after an automatic physical reset.

Current eval calls `_step_env()` and then reads command, body velocity,
projected gravity, and root height from `base_env.scene["robot"].data`. The
Task037 wrapper can perform `reset_trial()` or an outer reset inside that step,
so done environments may already expose reset-pose state.

Add a pre-reset terminal-state contract at the nearest layer that still owns
the failure state. It may be carried in `extras` or another explicit return
field, but it must be per environment and distinguish normal samples from
terminal samples. The evaluator must use terminal values only for affected
rows and current post-step values for rows that did not reset.

## Acceptance

- A fake multi-trial env sets an unmistakably bad terminal velocity/root pose
  and an unmistakably good reset pose in the same step.
- Eval accumulation records the bad terminal values for reset rows and cannot
  report the good reset values as the failure-frame sample.
- Mixed batches preserve normal post-step values for non-reset rows.
- Fall, timeout, completion, and reset-reason counts remain unchanged.
- JSON records the metric source/schema so old and corrected files can be
  distinguished.
- Action statistics remain associated with the pre-step action and trial.
- Re-evaluate at least one known falling Task045/Task046 checkpoint on local
  RTX 4090, when the matching checkpoint/runtime/assets are available, and
  compare corrected velocity, gravity, and root-height fields with the old
  JSON; do not overwrite the old file.
- Tests cover both inner-trial reset and final outer reset.

## Log

- 2026-08-07 Opened from
  `tools/task037_multitrial_eval_checkpoint.py`: robot state is read after
  `_step_env()` while the Task037 wrapper performs auto-reset during `step()`.
- 2026-08-07 Impact is bounded: fall/reset counts come from reset metadata and
  remain meaningful; failure-frame state-quality fields are provisional.
- 2026-08-07 Added terminal metric extras/schema/mask to the explicit
  `Task037MultiTrialVecEnvWrapper` reset path and to the MJLab inner-reset
  controller path.
- 2026-08-07 Updated `task037_multitrial_eval_checkpoint.py` to override only
  terminal rows with pre-reset command/velocity/yaw/gravity/root-z values and
  to record `task037_trial_metric_accumulator_v2` source counts.
- 2026-08-07 Added a fake-env regression where the fall frame has bad velocity,
  gravity, and root height while reset immediately restores a good pose. Local
  targeted pytest evidence is recorded in subtask 006 and passed.
- 2026-08-07 Runtime target changed to local RTX 4090. Corrected checkpoint
  re-eval is blocked locally by missing MJLab/task modules, missing configured
  G1 asset, and no local known-falling checkpoint path supplied. No checkpoint
  or asset was downloaded.
- 2026-08-10 After the MJLab runtime became locally runnable, attempted a
  Task038 true-TXL multi-trial eval smoke using the newly trained tiny Task040
  checkpoint. The first run exposed a wrapper bug:
  `task038_true_txl_multitrial_eval_smoke.py` did not define the
  `final_window_s` argparse field required by `task037_multitrial_eval_checkpoint.run_eval()`.
- 2026-08-10 Added `--final-window-s` pass-through to the Task038 eval wrapper
  and covered it in `tests/test_task038_true_txl_multitrial_eval_smoke.py`.
  Fresh test evidence:
  `UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -m pytest -q -p
  no:cacheprovider tests/test_task038_true_txl_multitrial_eval_smoke.py` -> 20
  passed; final Task047 targeted matrix including Task037/Task038 tests -> 78
  passed.
- 2026-08-10 Fresh local RTX 4090 pipeline evidence:
  `task047_local_4090_task038_multitrial_eval_after_final_window_fix.json` ->
  `pass=true`, `pipeline_pass=true`, `metric_schema=task037_multitrial_eval_metrics_v2`,
  with `quality_claim=false`, `eval_claim=false`, `reproduction_claim=false`.
  The checkpoint was the tiny from-scratch Task040 smoke checkpoint
  `outputs/task040/sequence_txl_ppo_update_smoke/model_1.pt`, not an external
  quality checkpoint.

## Review

Status: local code/regression fixed, and a local RTX 4090 corrected
multi-trial eval pipeline smoke now passes with a newly trained tiny Task040
checkpoint.

This subtask does not change any pass threshold. If corrected metrics are
worse, the experiment conclusion must follow the corrected evidence rather
than preserve the previous pass rate. The 2026-08-10 eval is pipeline evidence
only; a known-falling quality checkpoint comparison is still required before
promoting corrected Task045/Task046 quality claims.
