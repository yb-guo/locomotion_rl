# Route

Task: task001-agent-setup

Goal: Verify the lightweight local scaffold still imports and tests.

Scope:

- `src/h200_locomotion_lab`
- `tests`

Verify:

- `$env:PYTHONPATH=(Resolve-Path src).Path; python -m h200_locomotion_lab.tools.inspect_agent`
- `$env:PYTHONPATH=(Resolve-Path src).Path; python -m pytest`

Environment:

- local only

No Hack:

- no skipping tests
- no simulator dependency install

Hardware:

- should run without GPU

# Log

- Ran `python -m h200_locomotion_lab.tools.inspect_agent` with `PYTHONPATH=src`.
- Ran `python -m pytest`.
- Result: 2 tests passed.
- Note: pytest on this Windows directory cannot write normal `.pytest_cache` and creates `pytest-cache-files-*`; this is ignored.

# Review

Result: passed
Syntax: Python import and tests pass
Hack: no simulator dependency install
Scope: local scaffold only
Efficiency: lightweight tests only
Hardware: no GPU required
Verify: inspect agent and pytest passed
Findings: pytest cache warning remains environment-specific
