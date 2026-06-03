# 015 True-TXL Checkpoint Eval-Load Smoke

## Route

Close only the next minimal Task038 loop after `013` and `014`: prove a
checkpoint produced by the Task038 true-TXL runner path can be loaded by the
same train or heldout true-TXL runner task id and used for a short policy-action
rollout.

This is checkpoint eval-load and rollout plumbing only. It is not quality eval,
not reproduction evidence, and not a superiority claim. JSON claim boundaries
must keep `quality_claim:false`, `training_claim:false`, `eval_claim:false`,
`reproduction_claim:false`, `superiority_claim:false`, and
`checkpoint_eval_load_smoke_only:true`.

## Minimal Closed Loop

- Add a CLI that accepts a required `--checkpoint`, required `--output-json`,
  one of the train/heldout Task038 true-TXL runner smoke task ids, small
  `--num-envs`, short `--steps`, seed/device, and expected runner/model/action
  dimensions.
- Preflight before importing MJLab: reject missing checkpoints, unrelated task
  ids, non-positive env counts, and non-positive step counts with structured
  JSON.
- Construct the MJLab env and `Task038TrueTxlMemoryK160Runner`, call
  `runner.load(checkpoint, load_cfg={"actor": True}, strict=True,
  map_location=device)`, get the inference policy, reset, and step policy
  actions through the runner env.
- Summarize checkpoint path, task, seed/device, runner/model/action dimensions,
  `load_returned`, policy action shape/finite state, observation finiteness,
  optional `txl_debug_before` / `txl_debug_after`, extra keys, step count, and
  wall time.
- Treat `txl_debug_before` / `txl_debug_after` as optional diagnostics. If a
  snapshot fails before the true-TXL cache has initialized, record
  `txl_debug_before_error` / `txl_debug_after_error` and continue checkpoint
  load and rollout.
- Add local tests that exercise defaults, train/heldout allow-list behavior,
  unrelated task rejection, missing checkpoint failure JSON, pass/fail gates,
  JSON writing, `--help`, and doc overclaim checks without importing MJLab.

## Evidence Gate

Local evidence must pass:

```powershell
$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_true_txl_checkpoint_eval_smoke.py
```

H200 evidence is router-owned and requires at least one JSON from a checkpoint
created by the Task038 true-TXL runner path showing:

- task is either `Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke` or
  `Unitree-G1-Gripper-Flat-Task038-HeldoutTrueTxlRunnerSmoke`;
- `checkpoint_exists=true`;
- `runner_cls=Task038TrueTxlMemoryK160Runner`;
- `actor_model_class=Task038TrueTxlMemoryModel`;
- `action_dim=31` and `total_action_dim=31`;
- `load_returned=true`;
- at least one policy env step returned;
- `policy_action_shape=[actual_num_envs,31]`;
- `policy_action_finite=true`;
- finite non-empty observation summary;
- `policy_error=null`;
- `wall_time_s` recorded;
- `failure_reasons=[]`;
- all claim flags false and `checkpoint_eval_load_smoke_only=true`.

This gate does not compare metrics, evaluate heldout quality, or promote any
Task038 reproduction/superiority claim.

## Subagent Ownership

Worker `Task038/015` owns only:

- this document;
- the minimal `task.md` planned slice, loop, log, and status updates for `015`;
- `src/h200_locomotion_lab/tools/task038_true_txl_checkpoint_eval_smoke.py`;
- `tests/test_task038_true_txl_checkpoint_eval_smoke.py`.

Do not touch `.test_tmp_task021/`. Do not run H200 locally. Do not commit. Do
not download assets, checkpoints, datasets, simulator assets, or upstream repos.

## Failure Exit

If the checkpoint is absent, the task id is not a Task038 true-TXL runner smoke
task, the runner/model/action dimensions mismatch, strict actor load fails, or
the policy rollout cannot return at least one finite `[actual_num_envs,31]`
action, write the failure JSON and leave this slice pending for router/H200
diagnosis.

Do not change the failure into a policy-quality, training-progress, eval,
reproduction, or superiority result.

## Log

- 2026-05-30 Added the Task038 true-TXL checkpoint eval-load smoke CLI and local
  tests. The local gate covers defaults, task allow-listing, missing checkpoint
  preflight, structured failure summaries, pass/fail gate behavior, JSON
  writing, help, and no-overclaim docs without importing MJLab. H200 execution
  is pending router run; no quality/eval/reproduction/superiority claim is
  made.
- 2026-05-30 Local evidence gate passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_true_txl_checkpoint_eval_smoke.py`
  returned `14 passed in 0.11s` after the optional `txl_debug` snapshot fix.
  Router regression checks also passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp .test_tmp_task038_015_fix_router tests\test_task038_true_txl_checkpoint_eval_smoke.py tests\test_task038_true_txl_ppo_update_smoke.py tests\test_task038_true_txl_runner_smoke.py tests\test_task038_true_txl_inference_cache_safety.py tests\test_agent_inventory.py`
  returned `48 passed in 0.18s`, and the full Task038 regression set returned
  `154 passed in 0.78s`.
- 2026-05-30 Earlier H200 train eval-load smoke failed before rollout because
  `txl_debug_before` asserted before cache initialization. After making
  `txl_debug_before` / `txl_debug_after` optional diagnostics, router reran
  train and heldout checkpoint eval-load smokes on H200. Both returned
  `pass=true`, `load_returned=true`, `step_count=10`, `policy_action_shape=[8,31]`,
  `policy_action_finite=true`, `policy_error=null`, finite actor/history/critic
  observation summaries, `failure_reasons=[]`, and all claim flags false.
  Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_checkpoint_eval_load/train_true_txl_model0_eval_load_env8_steps10_optional_txl_debug.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_checkpoint_eval_load/heldout_true_txl_model0_eval_load_env8_steps10_optional_txl_debug.json`.
- 2026-05-30 Pre-fix regression checks also passed for
  `tests\test_task038_true_txl_ppo_update_smoke.py` (`16 passed in 0.09s`) and
  `tests\test_task038_true_txl_runner_smoke.py` (`12 passed in 0.05s`).
- 2026-05-30 Fixed H200 eval-load smoke failure mode where
  `txl_debug_before` could assert before true-TXL cache initialization. The CLI
  now records `txl_debug_before_error` / `txl_debug_after_error` for optional
  diagnostic failures and keeps the pass gate focused on checkpoint load,
  finite `[actual_num_envs,31]` policy action, finite observations, executed
  steps, and false claim flags.

## Review

Status: closed for train and heldout checkpoint eval-load smoke. The local
contract and post-fix regression tests passed, reviewer found no blocking
issues, and the H200 evidence JSON above shows the Task038 true-TXL `model_0.pt`
checkpoint can strict-load into both train and heldout true-TXL runner task ids
and execute a short finite policy rollout. This remains plumbing evidence only,
not a quality eval, reproduction result, or superiority claim.
