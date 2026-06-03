# 006: Retry-Context Consumer Smoke

## Route

Subtask 005 added a default-off actor-visible retry context. This subtask is
the first H200 consumer smoke for that contract.

Do not treat this as a quality-improvement run yet. The goal is to prove the
training pipeline can instantiate the actor with the enlarged actor-history
input, step PPO, write a checkpoint, and record wrapper debug evidence.

Important boundary:

- Enabling `task046_retry_context` appends six features to each actor frame.
- Therefore previous Stage2 checkpoints are not strict-load compatible unless
  an explicit migration path is added.
- First smoke starts from fresh initialization or an explicit shape-aware
  migration, not direct strict resume from Stage2.

Initial H200 command shape uses the Task044 hidden-fault train CLI. The clean
Task041 CLI parses the flag, but its clean runner does not install the
retry-context wrapper.

```bash
PYTHONPATH=src python -m h200_locomotion_lab.tools.task044_hidden_fault_train \
  --task046-retry-context \
  --task046-post-reset-recovery-reward \
  --task046-early-velocity-weight 1.2 \
  --task046-tail-velocity-weight 0.15 \
  --task046-orientation-weight 0.25 \
  --num-envs 64 \
  --rollout-steps 32 \
  --iterations 1 \
  --save-interval 1 \
  --seed 4620602 \
  --device cuda:0 \
  --output-json /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_context_consumer_smoke/smoke_train_summary.json \
  --log-dir /mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_context_consumer_smoke/logs
```

If smoke passes, the next closed unit may decide between:

- short fresh retry-context train;
- shape-aware Stage2 migration;
- or a policy-specific consumer change.

## Acceptance

- Local tests still pass:
  `tests/test_task041_sequence_txl_clean_train.py` and
  `tests/test_task044_hidden_fault_target.py`.
- `inspect_agent` passes.
- `task044_hidden_fault_train --help` exposes `--task046-retry-context`.
- H200 smoke JSON exists.
- H200 smoke has `train_pipeline_pass=true`.
- H200 smoke records non-null `task046_retry_context_debug` with
  `feature_dim=6`.
- H200 smoke writes a checkpoint.
- Review explicitly states whether this is only pipeline evidence or also a
  quality/eval claim.

## Log

- 2026-06-02 Opened after subtask 005 closed the local retry-context contract.
- 2026-06-02 Local validation before H200:
  - `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_task041_sequence_txl_clean_train.py tests/test_task044_hidden_fault_target.py --tb=short --basetemp .test_tmp_task046_consumer_smoke_local`
    passed: `22 passed, 8 skipped in 0.64s`.
  - `PYTHONPATH=src python -m h200_locomotion_lab.tools.inspect_agent`
    passed.
  - `PYTHONPATH=src python -m h200_locomotion_lab.tools.task044_hidden_fault_train --help`
    exposes `--task046-retry-context`.
- 2026-06-02 Corrected the route after review: the clean Task041 runner parses
  the flag but does not install `Task046RetryContextVecEnvWrapper`; the Task044
  hidden-fault train runner is the actual consumer.
- 2026-06-02 H200 local validation after syncing the run copy:
  `PYTHONPATH=src /mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python -m pytest -q -p no:cacheprovider tests/test_task041_sequence_txl_clean_train.py tests/test_task044_hidden_fault_target.py --tb=short --basetemp /tmp/task046_consumer_smoke_pytest`
  passed: `29 passed in 3.21s`.
- 2026-06-02 First H200 smoke failed before learning:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_context_consumer_smoke/smoke_train_summary.json`.
  Failure: `AttributeError("'dict' object has no attribute 'to'")` from
  `runner.learn()` because `Task046RetryContextVecEnvWrapper` converted the
  MJLab observation container into a plain dict. This is an interface bug, not
  a policy-quality failure.
- 2026-06-02 Fixed the wrapper to preserve observation containers that expose a
  `.to()` method, while still shallow-copying plain dict observations to avoid
  repeated context accumulation. Added a local regression test for this
  boundary.
- 2026-06-02 Local validation after the interface fix:
  `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_task041_sequence_txl_clean_train.py tests/test_task044_hidden_fault_target.py --tb=short --basetemp .test_tmp_task046_consumer_smoke_local3`
  passed: `22 passed, 8 skipped in 0.64s`; `inspect_agent` passed.
- 2026-06-02 H200 validation after syncing the fix:
  `PYTHONPATH=src /mnt/workspace/users/guoyubo/conda_envs/unitree-rl-mjlab/bin/python -m pytest -q -p no:cacheprovider tests/test_task041_sequence_txl_clean_train.py tests/test_task044_hidden_fault_target.py --tb=short --basetemp /tmp/task046_consumer_smoke_pytest2`
  passed: `30 passed in 2.88s`.
- 2026-06-02 Fixed H200 smoke passed:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_context_consumer_smoke/smoke_train_summary_fixed.json`.
  Result: `train_pipeline_pass=true`, `task044_train_pipeline_pass=true`,
  `learn_returned=true`, `checkpoint_exists=true`, `failure_reasons=[]`.
  Checkpoint:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task046/retry_context_consumer_smoke/logs_fixed/model_0.pt`.
  Retry-context debug: `feature_dim=6`, `base_actor_dim=104`, actor obs shape
  `[64, 110]`, `task046_retry_context` obs shape `[64, 6]`,
  `actor_history` shape `[64, 22560]`.

## Review

Status: evidence complete for retry-context consumer smoke.

The Task044 hidden-fault train path now consumes `task046_retry_context` and can
run one PPO iteration on H200 with the enlarged actor-history input. This is
pipeline evidence only: `quality_claim=false` and `quality_gate_pass=false` in
the smoke JSON. No eval quality or recovery improvement is claimed.

The first smoke exposed a real wrapper/container bug and was kept as diagnostic
evidence. The fixed smoke proves the interface boundary is now closed for the
minimal consumer path.
