# 001: Dynamic Train Entrypoint

## Route

Build the smallest Task043 training entrypoint by reusing Task041 train code.
The new code must only change the task id, output defaults, and summary labels;
the runner, actor, PPO algorithm, and trainable-scope guardrails stay shared.

## Acceptance

- `task043_dynamic_switch_train.py --help` works.
- Defaults point at
  `Unitree-G1-Gripper-Flat-Task043-TrainTrueTxlDynamicSwitchMemoryRequired-Fast1p6`.
- Registry patcher inserts the Task043 task with
  `runner_cls=Task038TrueTxlMemoryK160Runner`.
- Tests cover defaults, preflight, registry, and no-overclaim summary flags.

## Log

- 2026-05-31 Added the Task043 train wrapper and registry patcher.
- 2026-05-31 Local verification:
  - `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task043_dynamic_switch_train --help`
    passed;
  - `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task043_dynamic_switch_training_contract.py tests\test_task042_memory_ablation_contract.py tests\test_agent_inventory.py --tb=short`
    -> `16 passed`;
  - broader eval-parser regression with elevated shell permissions for pytest
    temp-dir creation -> `21 passed`.

## Review

Status: passed for local entrypoint contract. H200 smoke evidence is in subtask
002.
