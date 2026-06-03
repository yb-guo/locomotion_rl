# 014 True-TXL Inference Cache Safety Smoke

## Route

Close the next smallest Task038 boundary after `013`: prove the true-TXL actor
cache can be written safely after cache tensors were created or preserved under
PyTorch inference mode.

This is a local cache mutation safety smoke only. It makes no quality, eval,
training-progress, reproduction, or superiority claim:
`quality_claim:false`, `training_claim:false`, `eval_claim:false`,
`reproduction_claim:false`, `superiority_claim:false`, and
`inference_cache_safety_smoke_only:true`.

The H200 train-only PPO update in `013` already returned from
`runner.learn()`, but its optional post-learn `policy(obs)` diagnostic recorded
`Inplace update to inference tensor outside InferenceMode`. This slice fixes
that boundary before router reruns the real H200 policy-forward smoke.

## Minimal Closed Loop

- Keep Task038 true-TXL cache semantics unchanged: rollout/inference env-batch
  forwards append segment memory, inner reset records an event while preserving
  memory, and outer/full reset clears selected memory.
- Before any cache/counter tensor is mutated, replace PyTorch inference tensors
  with normal clones. The protected mutation points are layer memory append,
  memory lengths, reset counters, incremental step counters, segment counters,
  and token counters.
- Add local fake-tensor tests that simulate inference tensors raising on
  in-place writes outside inference mode. The tests prove repeated direct
  `_append_layer_memory()` grows memory lengths and reset hooks clear/record
  counters without that RuntimeError.
- Do not change the `013` PPO update gate or turn post-learn policy diagnostics
  into eval evidence.

## Evidence Gate

Local evidence must pass:

```powershell
$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_true_txl_inference_cache_safety.py
```

The local gate must show:

- inference cache tensors are cloned before mutation;
- repeated append reaches capped memory lengths without in-place inference
  tensor errors;
- inner reset, outer reset, and clear-memory hooks clone mutable counters and
  lengths before writes;
- docs keep all claim flags false and mark only
  `inference_cache_safety_smoke_only:true`.

H200 evidence remains router-owned. A router rerun may use the `013` train-only
PPO update smoke or a dedicated policy-forward smoke. A passing H200
policy-forward smoke only closes this cache-safety boundary; it does not claim
H200 eval, policy quality, reproduction, or superiority.

## Subagent Ownership

Worker `Task038/014` owns only:

- this document;
- the minimal `task.md` planned slice, loop, log, and status updates for `014`;
- `src/h200_locomotion_lab/training/rsl_history_wrapper.py`;
- `tests/test_task038_true_txl_inference_cache_safety.py`.

Do not touch `.test_tmp_task021/`. Do not run H200 locally. Do not download
assets, checkpoints, datasets, simulator assets, or upstream repos.

## Failure Exit

If local fake tests cannot close the inference-cache mutation boundary, leave
`014` pending and record the failing mutation point. Do not broaden scope into
heldout eval, long training, quality metrics, or checkpoint work.

If the H200 router later still sees the same PyTorch inference-tensor mutation
RuntimeError in a true policy forward, keep the failure as Task038/014 follow-up
evidence rather than marking eval or reproduction blocked.

## Log

- 2026-05-30 Added clone-before-mutation handling for Task038 true-TXL actor
  cache tensors and counters. Local fake-tensor tests simulate PyTorch
  inference tensors that reject in-place writes outside inference mode, and
  cover repeated layer-memory append plus inner/outer/full reset mutation
  paths. No H200 run, eval, quality, reproduction, or superiority claim is
  made.
- 2026-05-30 Local evidence gate passed:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_true_txl_inference_cache_safety.py`
  returned `4 passed in 0.06s`. Regression checks also passed for
  `tests\test_task038_true_txl_reset_hook.py` (`9 passed in 0.06s`) and
  `tests\test_task038_true_txl_ppo_update_smoke.py` (`16 passed in 0.08s`).
- 2026-05-30 Router reran the real H200 Task038 train-only true-TXL PPO update
  smoke after syncing the cache-safety fix. The post-learn policy action
  forward now returned `policy_action_shape=[8,31]`,
  `policy_action_finite=true`, and `policy_error=null`; the full probe also
  returned `pass=true`, `learn_returned=true`, `failure_reasons=[]`,
  `txl_debug.stateless_forward_batches=1`, and
  `txl_debug.stateless_forward_samples=16`. This closes only the H200
  policy-forward cache-safety boundary, not eval quality or reproduction.
  Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_inference_cache_safety/train_true_txl_ppo_update_env8_iter1_policy_forward_safe.json`.

## Review

Status: closed for the local and H200 inference-cache safety smoke. The local
fake-tensor gate verifies clone-before-mutation for the direct layer-memory
append path and reset hook mutation paths. Reviewer found no blocking issues,
and the H200 evidence above shows the real post-learn policy forward no longer
hits the PyTorch inference-tensor mutation error. This slice makes no eval,
quality, reproduction, or superiority claim.
