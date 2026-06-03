# 001: Hidden-Fault Eval Contract

## Route

Task043's failure mode is a target-design failure, not a simulator plumbing
failure. The next target needs a local, simulator-free contract before any H200
run can be called useful.

Implement a pure triplet evaluator:

- input: normal, zero-residual, and stateless-memory eval summaries;
- require hidden-fault metadata proving the actor did not receive direct fault
  labels;
- require normal mode to pass pipeline and locomotion quality;
- treat ablations as records even when their top-level pipeline is false due to
  intentional stateless debug disablement;
- require both ablations to degrade materially by final-trial metrics.

## Acceptance

- Tests cover positive degraded triplet.
- Tests cover tied ablations failing the Task044 gate.
- Tests cover normal quality failure.
- Tests cover missing/visible fault metadata.
- Tests cover stateless ablation accepted as an ablation record even when its
  top-level pipeline flag is false.

## Log

- 2026-05-31 Started as the first Task044 closed unit.
- 2026-05-31 Added
  `src/h200_locomotion_lab/training/task044_memory_required_contract.py` and
  `tests/test_task044_memory_required_contract.py`.
- 2026-05-31 Local validation passed as part of:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider tests\test_task044_memory_required_contract.py tests\test_task044_triplet_summary_cli.py tests\test_agent_inventory.py --tb=short --basetemp pytest_tmp_task044_contract3`
  with 11 passed.

## Review

Status: passed.

The contract closes the Task043 loophole: normal walking quality alone is not
enough. A triplet only passes when normal quality is good and both
zero-residual and stateless-memory ablations materially degrade. Stateless
top-level pipeline false is tolerated as an ablation record when the ablation
mode is explicit.
