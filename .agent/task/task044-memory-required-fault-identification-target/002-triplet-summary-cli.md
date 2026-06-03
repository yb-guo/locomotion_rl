# 002: Triplet Summary CLI

## Route

After the pure contract is locked, add a thin CLI that reads three H200 eval
JSON files and writes one Task044 triplet summary. The CLI should not run
MuJoCo; it only evaluates evidence already produced by Task043/Task044 eval
commands.

## Acceptance

- CLI `--help` works locally.
- Tests prove source JSON paths are recorded.
- Tests prove no-overclaim fields remain false.
- CLI requires explicit `--confirm-hidden-fault-labels` to annotate JSON files
  that do not already contain Task044 hidden-fault metadata.
- Missing JSON or malformed triplets fail with explicit reasons.

## Log

- 2026-05-31 Planned after 001.
- 2026-05-31 Added
  `src/h200_locomotion_lab/tools/task044_memory_required_triplet_summary.py`
  and `tests/test_task044_triplet_summary_cli.py`.
- 2026-05-31 Local validation passed:
  `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.task044_memory_required_triplet_summary --help`.
- 2026-05-31 The pytest slice covering 001/002 passed with 11 tests.

## Review

Status: passed.

The CLI is intentionally a JSON comparator, not another simulator runner. It
requires explicit `--confirm-hidden-fault-labels` before annotating summaries
that do not already contain Task044 hidden-fault metadata.
