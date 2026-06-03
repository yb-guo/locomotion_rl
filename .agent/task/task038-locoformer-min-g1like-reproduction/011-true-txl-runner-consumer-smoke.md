# 011 True-TXL Runner Consumer Smoke

## Route

Close only the minimum Task038 runner-consumer wiring loop that distinguishes a
stateful true-TXL cache consumer from the earlier `010` segment-pooling runner
smoke.

This slice registers separate external MJLab task ids:

- `Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke`;
- `Unitree-G1-Gripper-Flat-Task038-HeldoutTrueTxlRunnerSmoke`.

Those ids reuse the Task038 train/held-out XML env cfg helpers, but use
`Task038TrueTxlMemoryK160Runner`, which installs the same deterministic inner
reset controller and K160 history wrapper route as the Task037 deterministic
runner while setting the actor class to `Task038TrueTxlMemoryModel`.

This is a runner smoke only. It makes `quality_claim:false`,
`training_claim:false`, `eval_claim:false`, and `runner_smoke_only:true`.

## Minimal Closed Loop

- Add `Task038TrueTxlMemoryModel` without changing
  `Task037TxlStyleMemoryModel` semantics.
- Add `Task038TrueTxlMemoryK160Runner` without changing
  `Task037TxlMemoryK160DeterministicRunner` semantics.
- Extend the external MJLab patcher with true-TXL smoke task ids and idempotent
  imports.
- Add a probe that constructs env/runner, does not load checkpoints, does not
  train, forwards the policy at least twice, and steps zero actions through
  `runner.env`.
- Gate pass on the second policy forward showing positive previous-memory cache
  exposure in `txl_debug`. This records that valid previous segment tokens were
  available to the attention key/value path; it does not claim useful attention
  weights or locomotion quality.

## Evidence Gate

Local evidence must pass:

```powershell
$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider --basetemp .test_tmp_task038_011 tests\test_task038_true_txl_runner_smoke.py tests\test_task038_txl_memory_contract.py
```

H200 evidence requires train and held-out true-TXL runner probes to write JSON
showing:

- `runner_cls=Task038TrueTxlMemoryK160Runner`;
- `actor_model_class=Task038TrueTxlMemoryModel`;
- `action_dim=31` and `total_action_dim=31`;
- finite policy action shape `[actual_num_envs,31]`;
- non-empty finite observation summary;
- `step_required_extras_missing=[]`;
- valid `txl_debug.envs` entries with `env_id`, `memory_lengths`,
  `inner_reset_events`, `outer_reset_events`, and `incremental_steps`;
- at least one env with positive `last_attended_previous_memory_lengths` after
  repeated policy forward, interpreted only as previous-memory exposure.

## Subagent Ownership

Worker `Task038/011` owns only:

- this document;
- the minimal `task.md` status append for `011`;
- the Task038 external MJLab patcher true-TXL task id/import extension;
- `Task038TrueTxlMemoryModel` and `Task038TrueTxlMemoryK160Runner` additions in
  `src/h200_locomotion_lab/training/rsl_history_wrapper.py`;
- `src/h200_locomotion_lab/tools/task038_true_txl_runner_smoke_probe.py`;
- `tests/test_task038_true_txl_runner_smoke.py`.

Do not touch `.test_tmp_task021/`. Do not start training, eval, checkpoint
load/save, or PPO update.

## Failure Exit

If external MJLab or RSL runner APIs prevent exact reset-hook integration, stop
at cache-forward smoke evidence and record reset integration as pending. Do not
fake reset events. The minimum pass gate still requires repeated policy forward
to prove current segment exposure to previous cached memory.

If H200 runner construction fails, record the failure JSON and leave this slice
pending. Do not turn this into a policy-quality, training, eval, reproduction,
or TXL superiority claim.

## Log

- 2026-05-29 Added local Task038/011 true-TXL runner-consumer smoke wiring:
  stateful per-env cache actor, K160 runner class, true-TXL MJLab task ids,
  no-training probe, and local pass-gate tests. H200 evidence is pending.
- 2026-05-29 Tightened the smoke claim from previous-memory attention quality to
  previous-memory cache exposure after reviewer flagged that attention weights
  are not recorded.
- 2026-05-29 Ran the H200 true-TXL runner-consumer smoke on `cuda:0` for train
  and held-out Task038 XML variants. Both env8 probes passed with
  `runner_cls=Task038TrueTxlMemoryK160Runner`,
  `actor_model_class=Task038TrueTxlMemoryModel`, `action_dim=31`,
  `total_action_dim=31`, finite `[8,31]` policy action, finite
  actor/actor_history/critic observations, `step_required_extras_missing=[]`,
  `failure_reasons=[]`, and positive previous-memory cache exposure after the
  repeated policy forward. Evidence JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_runner_smoke/train_true_txl_runner_smoke_env8.json`
  and
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_runner_smoke/heldout_true_txl_runner_smoke_env8.json`.

## Review

Status: closed by final reviewer confirmation for runner-consumer smoke only.

This slice proves only no-training runner construction, policy forward,
zero-action stepping, and previous-memory cache exposure in the true-TXL actor.
It does not claim reset-hook integration in the real training loop, locomotion
quality, training progress, evaluation success, reproduction, or TXL
superiority.
