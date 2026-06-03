# 016 True-TXL Multitrial Metric Eval Smoke

## Route

Close only the next minimal Task038 loop after `015`: prove the multi-trial
metric eval pipeline can run against a Task038 true-TXL checkpoint and emit
comparable per-trial JSON for the train or heldout true-TXL runner smoke task
ids.

This is eval pipeline plumbing only. It is not a policy quality conclusion, not
reproduction evidence, and not a superiority claim. The top-level
`pass`/`eval_pipeline_smoke_pass` gate means the JSON pipeline completed and
produced finite comparable trial metrics. The Task037-style final-trial quality
field is preserved only as `quality_metric_final_trial_pass`.

JSON claim boundaries must keep `quality_claim:false`, `training_claim:false`,
`eval_claim:false`, `reproduction_claim:false`, `superiority_claim:false`, and
`eval_pipeline_smoke_only:true`.

## Minimal Closed Loop

- Add a CLI that accepts a required `--checkpoint`, required `--output-json`,
  one of the train/heldout Task038 true-TXL runner smoke task ids, small
  `--num-envs`, short `--steps`, short `--trial-length-s`, seed/device,
  velocity arguments, optional Task037 dynamic/dead-joint eval knobs, and
  expected runner/model/action dimension arguments.
- Preflight before importing MJLab through the delegated evaluator: reject
  missing checkpoints, unrelated task ids, non-positive env counts,
  non-positive step counts, and non-positive trial lengths with structured
  JSON.
- Reuse the existing Task037 multi-trial evaluator for the real runner/env
  construction, strict actor checkpoint load, rollout, and per-trial metric
  calculation.
- Wrap the evaluator output so Task038/016 top-level `pass`,
  `pipeline_pass`, and `eval_pipeline_smoke_pass` are controlled only by
  pipeline health: allowed task id, checkpoint field, `trial_0`, `final_trial`,
  `aggregate`, positive env/step counts, finite metric values, no top-level
  exception, and false claim flags.
- Preserve Task037 `final_trial_pass` as
  `quality_metric_final_trial_pass`/`task037_final_trial_pass`, and preserve
  Task037 top-level pass as `task037_pass`. Neither field may control the
  Task038/016 smoke pass.
- Add local tests that exercise defaults, train/heldout allow-list behavior,
  unrelated task rejection, missing checkpoint failure JSON, final-trial-fail
  but pipeline-pass behavior, trial field rejection, exception/nonfinite
  rejection, overclaim rejection, JSON writing, `--help`, and doc overclaim
  checks without importing MJLab.

## Evidence Gate

Local evidence must pass:

```powershell
$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_true_txl_multitrial_eval_smoke.py
```

H200 evidence is router-owned and requires at least one JSON from a checkpoint
created by the Task038 true-TXL runner path showing:

- task is either `Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke` or
  `Unitree-G1-Gripper-Flat-Task038-HeldoutTrueTxlRunnerSmoke`;
- `trial_0`, `final_trial`, and `aggregate` metric objects exist;
- metric values are finite and comparable JSON values;
- `quality_metric_final_trial_pass` exists as a metric-only field;
- `task037_pass` exists as the delegated evaluator's final-trial gate;
- top-level `pass=true`, `pipeline_pass=true`, and
  `eval_pipeline_smoke_pass=true` can be true even when
  `quality_metric_final_trial_pass=false`;
- `promotion_gate=pipeline_smoke_only`;
- `failure_reasons=[]`;
- all claim flags false and `eval_pipeline_smoke_only=true`.

This gate does not promote any Task038 quality, eval, reproduction, or
superiority claim, especially for a random or one-iteration checkpoint.

## Subagent Ownership

Worker `Task038/016` owns only:

- this document;
- the minimal `task.md` planned slice, loop, log, and status updates for `016`;
- `src/h200_locomotion_lab/tools/task038_true_txl_multitrial_eval_smoke.py`;
- `tests/test_task038_true_txl_multitrial_eval_smoke.py`.

Do not touch `.test_tmp_task021/`. Do not run H200 locally. Do not commit. Do
not download assets, checkpoints, datasets, simulator assets, or upstream repos.

## Failure Exit

If the checkpoint is absent, the task id is not a Task038 true-TXL runner smoke
task, strict actor load fails, the delegated evaluator raises, required trial
metric fields are absent, metric values are non-finite, or claim flags overstep
the smoke boundary, write the failure JSON and leave this slice pending for
router/H200 diagnosis.

Do not reinterpret `quality_metric_final_trial_pass=false` as a pipeline
failure. Do not reinterpret `quality_metric_final_trial_pass=true` as a Task038
policy-quality, eval, reproduction, or superiority result.

## Log

- 2026-05-30 Added the Task038 true-TXL multi-trial metric eval smoke CLI and
  local tests. The CLI reuses the Task037 multi-trial evaluator, then wraps the
  JSON so top-level pass gates only on pipeline health while preserving
  `final_trial_pass` as `quality_metric_final_trial_pass`. H200 execution is
  pending router run; no quality/eval/reproduction/superiority claim is made.
- 2026-05-30 Local evidence gate passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_true_txl_multitrial_eval_smoke.py`
  returned `13 passed in 0.11s`. Focused regression with the adjacent
  checkpoint eval-load smoke also passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_true_txl_multitrial_eval_smoke.py tests\test_task038_true_txl_checkpoint_eval_smoke.py`
  returned `27 passed in 0.15s`.
- 2026-05-30 Fixed P1 reviewer blocker in the pipeline pass gate: the gate now
  requires explicit trial and aggregate metric schema fields, rejects `None`,
  `NaN`, and `inf` for required numeric metrics, and requires
  `reset_reason_counts` to be a dict while preserving the decoupled
  `quality_metric_final_trial_pass` behavior. Local evidence gate passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_true_txl_multitrial_eval_smoke.py`
  returned `20 passed in 0.17s`.
- 2026-05-30 Router post-fix local verification passed after the schema gate
  fix. Focused regression:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp .test_tmp_task038_016_fix_router tests\test_task038_true_txl_multitrial_eval_smoke.py tests\test_task038_true_txl_checkpoint_eval_smoke.py tests\test_task038_true_txl_inference_cache_safety.py tests\test_agent_inventory.py`
  returned `40 passed in 0.20s`. Full Task038 regression:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp .test_tmp_task038_all_016_fix_router tests\test_task038_claim_contract.py tests\test_task038_eval_contract.py tests\test_task038_g1like_mjcf_patch.py tests\test_task038_g1like_morphology_manifest.py tests\test_task038_g1like_slot_contract.py tests\test_task038_mjlab_runner_smoke.py tests\test_task038_mjlab_variant_env_load.py tests\test_task038_true_txl_checkpoint_eval_smoke.py tests\test_task038_true_txl_inference_cache_safety.py tests\test_task038_true_txl_multitrial_eval_smoke.py tests\test_task038_true_txl_ppo_update_smoke.py tests\test_task038_true_txl_reset_hook.py tests\test_task038_true_txl_runner_smoke.py tests\test_task038_txl_memory_contract.py tests\test_agent_inventory.py`
  returned `174 passed in 1.13s`. `python -m h200_locomotion_lab.tools.inspect_agent`
  also completed successfully.
- 2026-05-30 Review subagent found no blockers for the post-fix `016` schema
  gate and claim-boundary docs. Residual risk: the smoke schema gate validates
  `trial_0`, `final_trial`, and `aggregate`, but does not schema-check every
  intermediate `trial_N`.
- 2026-05-30 Ran the `016` H200 metric pipeline smoke on `cuda:0` for both the
  train and heldout Task038 true-TXL runner task ids using the Task038-generated
  `model_0.pt` from the `014` smoke. Both JSON files returned `pass=true`,
  `pipeline_pass=true`, `eval_pipeline_smoke_pass=true`,
  `quality_metric_final_trial_pass=false`, `task037_pass=false`,
  `promotion_gate=pipeline_smoke_only`, `failure_reasons=[]`, and all claim
  flags false. This closes only the metric pipeline smoke; it does not promote a
  policy-quality, eval, reproduction, or superiority claim. Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_multitrial_eval_smoke/train_true_txl_model0_multitrial_metric_env8_steps60_schema_gate.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_multitrial_eval_smoke/heldout_true_txl_model0_multitrial_metric_env8_steps60_schema_gate.json`.

## Review

Status: local implementation, post-fix local verification, independent review,
and router-owned H200 train/heldout metric pipeline smoke evidence are complete.
This slice is closed as pipeline plumbing only. No quality, reproduction, or
superiority claim is made.
