# 012 True-TXL Reset-Hook Integration Smoke

## Route

Close only the next minimal Task038 loop after `011`: wire the true-TXL actor
reset hooks into the runner-visible `env.step()` and `env.reset()` path.

This slice keeps the existing Task037 multi-trial semantics intact and adds a
Task038-only reset-hook wrapper around the Task038 true-TXL runner env. The
wrapper dispatches multi-trial extras to `Task038TrueTxlMemoryModel`:

- `inner_reset` / `task037_inner_reset` records an inner reset and preserves
  selected env memory lengths;
- `outer_reset` / `episode_done` / `task037_outer_reset` records an outer reset
  and clears selected env memory lengths;
- full `env.reset()` clears any existing actor cache for all envs.

This slice proves reset-hook integration smoke only and makes no
training/eval/quality/reproduction/superiority claim. Probe JSON must keep
`quality_claim:false`, `training_claim:false`, `eval_claim:false`,
`reproduction_claim:false`, `superiority_claim:false`, and
`reset_hook_integration_smoke_only:true`.

## Minimal Closed Loop

- Add a Task038-only runner env wrapper that intercepts reset extras after
  Task037 multi-trial stepping and before the next policy forward.
- Attach the constructed true-TXL actor to that wrapper after
  `Task038TrueTxlMemoryK160Runner` calls `super().__init__`.
- Add a no-training probe that constructs the true-TXL runner, performs policy
  forward plus `env.step()` loops with short episode-length support, and writes
  JSON evidence for runner/model/action dims, reset extras, actor reset counters,
  inner memory preservation, outer memory clearing, and claim-boundary flags.
- Add local tests with fake env/actor and pass-gate fixtures that reject missing
  reset events, inner reset clearing memory, outer reset failing to clear
  memory, wrong runner/model, overclaim flags, and doc overclaims.

## Evidence Gate

Local evidence must pass:

```powershell
$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp .test_tmp_task038_012 tests\test_task038_true_txl_reset_hook.py tests\test_task038_true_txl_runner_smoke.py tests\test_task038_txl_memory_contract.py
```

H200 reset-hook smoke requires train and held-out JSON showing:

- `runner_cls=Task038TrueTxlMemoryK160Runner`;
- `actor_model_class=Task038TrueTxlMemoryModel`;
- `action_dim=31` and `total_action_dim=31`;
- finite policy action shape `[actual_num_envs,31]`;
- `step_required_extras_missing=[]`;
- `saw_inner_reset=true` and `saw_outer_reset=true`;
- positive actor inner and outer reset event totals;
- `inner_reset_preserved_memory_before_next_policy=true`;
- `outer_reset_cleared_memory_before_next_policy=true`;
- all claim flags false and `reset_hook_integration_smoke_only=true`.

## Subagent Ownership

Worker `Task038/012` owns only:

- this document;
- the minimal `task.md` status append for `012`;
- Task038 reset-hook wiring in
  `src/h200_locomotion_lab/training/rsl_history_wrapper.py`;
- `src/h200_locomotion_lab/tools/task038_true_txl_reset_hook_probe.py`;
- `tests/test_task038_true_txl_reset_hook.py`.

Do not touch `.test_tmp_task021/`. Do not start training, eval, checkpoint
load/save, or PPO updates. Do not download assets, checkpoints, datasets,
simulator assets, or upstream repos.

## Failure Exit

If external MJLab or RSL runner APIs prevent actor hook attachment, stop with
local fake-env evidence and leave H200 reset-hook integration pending. Do not
fake reset events in probe summaries.

If H200 runner construction or short reset stepping fails, write the failure
JSON and leave this slice pending. Do not convert the failure into a training,
eval, quality, reproduction, or TXL superiority claim.

## Log

- 2026-05-29 Added Task038-only reset-hook wrapper wiring, a no-training
  reset-hook probe, and local tests for reset hook dispatch and false-pass
  gates. H200 reset-hook integration evidence is pending router execution.
- 2026-05-29 Ran the H200 reset-hook integration smoke on `cuda:0` for train
  and held-out Task038 true-TXL runner tasks. Both env8 probes passed with
  `runner_cls=Task038TrueTxlMemoryK160Runner`,
  `actor_model_class=Task038TrueTxlMemoryModel`, `action_dim=31`,
  `total_action_dim=31`, finite `[8,31]` policy action,
  `step_required_extras_missing=[]`, `saw_inner_reset=true`,
  `saw_outer_reset=true`, positive actor reset event totals,
  `inner_reset_preserved_memory_before_next_policy=true`,
  `outer_reset_cleared_memory_before_next_policy=true`, `failure_reasons=[]`,
  and all claim flags false. Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_reset_hook/train_true_txl_reset_hook_env8.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_reset_hook/heldout_true_txl_reset_hook_env8.json`.

## Review

Status: closed by final reviewer confirmation for reset-hook integration smoke only.

This slice proves only reset-hook integration timing and true-TXL cache
preserve/clear semantics in a no-training runner smoke, and makes no training/eval/quality/reproduction/superiority claim.
