# 005 TXL-Style Memory Consumer

## Route

Add a long-context policy consumer only after the multi-trial contract and eval
semantics are stable.

Acceptance criteria:

- Target context horizon is `3.2s = 160` policy steps at `50Hz`.
- TXL-style memory preserves state across inner trial reset.
- TXL-style memory clears state on outer episode reset.
- Actor does not see trial labels or failure debug labels.
- env64 inference and one PPO iteration run.
- env8192 overhead is recorded.
- No locomotion-quality claim is made from construction smoke.

## Log

- 2026-05-29 Planned.
- 2026-05-29 Added `Task037TxlStyleMemoryModel` and
  `Task037TxlMemoryK160DeterministicRunner`.
  - Context horizon: `160` policy steps = `3.2s` at `50Hz`.
  - Actor history frame dim: `135` (`104` actor obs + `31` action).
  - Actor history input dim: `21600`.
  - Segment memory: `segment_len=16`, `segment_count=10`,
    `token_dim=128`, token projection `Linear(135, 128)`.
  - The model consumes only actor-visible history and does not receive trial
    labels, failure id, motor scale, or failure mask.
- 2026-05-29 Added H200 task id
  `Unitree-G1-Gripper-Flat-Task037-TxlMemoryK160-DeterministicInnerReset-Fast2p0`.
- 2026-05-29 Added fake-env regression coverage for K160 memory reset:
  inner trial reset preserves history, outer episode reset clears history.
- 2026-05-29 Local validation:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_task037_multitrial_contract.py tests/test_task037_mjlab_smoke_scripts.py tests/test_agent_inventory.py`
    -> `6 passed, 4 skipped in 0.09s`.
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m h200_locomotion_lab.tools.inspect_agent`
    -> passed.
- 2026-05-29 H200 validation:
  - `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter`,
    `tests/test_task037_multitrial_contract.py tests/test_task037_mjlab_smoke_scripts.py`
    -> `8 passed in 2.48s`.
  - Registry check resolved the task id to
    `Task037TxlMemoryK160DeterministicRunner`.
- 2026-05-29 H200 env64 one-iteration PPO smoke:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/txl_memory_smoke/037_txl_k160_env64_iter1_gpu0_seed3700501.stdout.log`.
  Result: `pass=true`, `steps_per_second=1423`, collection `0.918s`,
  learning `0.161s`.
- 2026-05-29 H200 env8192 overhead smoke:
  `/mnt/workspace/users/guoyubo/agent_workspace/task025_sonic_mjlab_adapter/outputs/task037/txl_memory_smoke/037_txl_k160_env8192_iter1_gpu0_seed3700511.stdout.log`.
  Result: `pass=true`, `steps_per_second=55595`, collection `3.065s`,
  learning `0.471s`.
- 2026-05-29 Local summary:
  `task037_txl_memory_smoke_summary.json`.

## Review

Status: passed for construction, memory reset, and overhead smoke.

This is not a full Transformer-XL implementation and not a locomotion-quality
claim. It proves the Task037 long-context consumer can be constructed, trained
for one PPO iteration, preserves K160 memory across inner resets, clears it on
outer resets, and has recorded env8192 overhead.
