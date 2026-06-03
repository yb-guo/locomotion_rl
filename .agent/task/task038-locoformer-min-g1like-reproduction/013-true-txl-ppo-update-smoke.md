# 013 True-TXL PPO Update Smoke

## Route

Close only the next minimal Task038 loop after `012`: prove the Task038 true-TXL
train runner can enter and return from one tiny PPO update path.

This is a training-path crash smoke only. It uses the train variant only:
`Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke`. It does no heldout training,
no long training, no checkpoint download, and no eval. The JSON claim
boundary must keep `quality_claim:false`, `training_claim:false`,
`eval_claim:false`, `reproduction_claim:false`, `superiority_claim:false`, and
`ppo_update_smoke_only:true`. This slice makes no quality/eval/reproduction/superiority claim.

## Minimal Closed Loop

- Add a train-only CLI smoke that constructs the MJLab env and
  `Task038TrueTxlMemoryK160Runner` through the registered Task038 true-TXL train
  runner task id.
- Mutate the loaded agent config only for the smoke: small `num_steps_per_env`,
  one learning iteration, tensorboard logging, no model upload, no resume, and
  conservative save interval.
- Call `runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)`.
- After `learn()` returns, summarize runner/model/action dims, log directory
  existence/files, `txl_debug`, wall time, and one inference policy action if
  observations are available.
- The Task038 true-TXL actor uses stateful env-batch cache for rollout/inference
  forwards. During flattened PPO update minibatches where the current batch size
  differs from the cache env count, it uses stateless current-segment attention
  and records `txl_debug.stateless_forward_batches` /
  `txl_debug.stateless_forward_samples`. This keeps the update crash-smoke path
  runnable but is not full sequence-aware TXL training; full sequence-aware TXL
  training remains a later task.
- Add local tests that exercise parse defaults, config mutation, pass gates,
  failure summaries, JSON writing, claim-boundary rejection, and doc overclaim
  checks without importing MJLab.

## Evidence Gate

Local evidence must pass:

```powershell
$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task038_true_txl_ppo_update_smoke.py
```

H200 PPO update smoke requires one train-only JSON showing:

- `task=Unitree-G1-Gripper-Flat-Task038-TrainTrueTxlRunnerSmoke`;
- `runner_cls=Task038TrueTxlMemoryK160Runner`;
- `actor_model_class=Task038TrueTxlMemoryModel`;
- `action_dim=31` and `total_action_dim=31`;
- `learn_returned=true`;
- `txl_debug.stateless_forward_batches > 0`;
- `iterations=1`;
- positive `num_envs` and `rollout_steps`;
- `log_dir_exists=true`;
- `wall_time_s` recorded;
- no top-level probe or `runner.learn()` exception and `failure_reasons=[]`;
- all claim flags false and `ppo_update_smoke_only=true`.

Policy action summary is opportunistic after `learn()` returns. If present, it
must be finite and shaped `[actual_num_envs,31]`; absence of the post-learn
policy action is not an eval failure because this gate is only the PPO update
return path.

## Subagent Ownership

Worker `Task038/013` owns only:

- this document;
- the minimal `task.md` status append for `013`;
- `src/h200_locomotion_lab/tools/task038_true_txl_ppo_update_smoke.py`;
- `tests/test_task038_true_txl_ppo_update_smoke.py`.

Do not touch `.test_tmp_task021/`. Do not run H200 locally. Do not use heldout
Task038 variants for this smoke. Do not download assets, checkpoints, datasets,
simulator assets, or upstream repos.

## Failure Exit

If MJLab, RSL runner storage, or Task038 true-TXL actor state prevents one tiny
`runner.learn()` call from returning, write the failure JSON and leave this
slice pending for router/H200 diagnosis. Do not convert a returned update smoke
into a training progress, eval, policy quality, reproduction, or superiority
claim.

If the train task id is missing or resolves to any runner/model other than
`Task038TrueTxlMemoryK160Runner` / `Task038TrueTxlMemoryModel`, stop at the
structured failure. Do not switch to the heldout task or a Task037 runner to get
a pass.

## Log

- 2026-05-29 Added the train-only true-TXL PPO update smoke CLI and local tests.
  The local gate covers defaults, config mutation, pass/fail summary logic, JSON
  writing, and claim-boundary docs without requiring MJLab. H200 execution is
  pending router run; no quality/eval/reproduction/superiority claim is made.
- 2026-05-30 Added the minimal PPO minibatch fallback contract: env-batch
  rollout/inference forwards keep using the stateful cache, while flattened PPO
  update minibatches with a batch size different from the env cache count use
  stateless current-segment attention and must record positive
  `txl_debug.stateless_forward_batches`. This is an update-path crash smoke only,
  not full sequence-aware TXL training and not quality/eval/reproduction
  evidence.
- 2026-05-30 H200 first exposed a second boundary issue after `runner.learn()`
  returned: the opportunistic post-learn `policy(obs)` summary hit PyTorch's
  inference-tensor in-place update guard. The probe now keeps post-learn policy
  action summary diagnostic-only, records `policy_error` when it fails, and
  gates the smoke on `learn_returned=true`, positive
  `txl_debug.stateless_forward_batches`, dimensions, log directory, and claim
  flags instead.
- 2026-05-30 Ran the train-only H200 PPO update smoke on `cuda:0`. The probe
  returned `pass=true`, `learn_returned=true`, `runner_cls=Task038TrueTxlMemoryK160Runner`,
  `actor_model_class=Task038TrueTxlMemoryModel`, `action_dim=31`,
  `total_action_dim=31`, `iterations=1`, `num_envs=8`, `rollout_steps=2`,
  `log_dir_exists=true`, `failure_reasons=[]`,
  `txl_debug.stateless_forward_batches=1`, and
  `txl_debug.stateless_forward_samples=16`. The optional post-learn policy
  action summary recorded the inference-tensor `policy_error` with
  `policy_action_shape=null`, which is diagnostic-only for this smoke. Evidence
  JSON:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task038/true_txl_ppo_update_smoke/train_true_txl_ppo_update_env8_iter1_postlearn_optional.json`.

## Review

Status: closed for the train-only PPO update smoke. Reviewer found no blocking
issues in the fallback and post-learn diagnostic handling, local regression
tests passed, and the H200 evidence JSON above satisfies this subtask's gate.

This slice proves only that the Task038 true-TXL train runner can execute one
tiny PPO update path without crashing once router records passing H200 JSON. It
makes no policy quality, eval, reproduction, or superiority claim.
