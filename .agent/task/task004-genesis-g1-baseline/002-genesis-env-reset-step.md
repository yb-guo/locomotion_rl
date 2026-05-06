# Route

Task: task004-genesis-g1-baseline

Goal: Implement or wrap a Genesis env that can reset and step.

Scope:

- `src/h200_locomotion_lab/envs/genesis_adapter.py`
- tests for reset/step boundary where possible

Verify:

- Minimal script resets and steps without training.

Environment:

- Linux H200 target for real Genesis
- local stub tests allowed

No Hack:

- no simulator import at module import time if it breaks local tests
- no global mutable singleton scene
- no unbounded per-step Python logging

Hardware:

- avoid CPU/GPU sync in hot path
- batch envs when real training starts

# Log

- 2026-05-06: Replaced the placeholder `GenesisG1Env` with a simulator-independent
  reset/step boundary in `src/h200_locomotion_lab/envs/genesis_adapter.py`.
- The module does not import `genesis` at module import time.
- Added:
  - `GenesisG1Contract`
  - `StepResult`
  - `GenesisBackend` protocol
  - `ContractOnlyBackend` for local boundary tests
  - `GenesisG1Env.contract_only()` for local reset/step verification
- Added local tests in `tests/test_genesis_adapter.py`.
- H200 check:

```bash
python3 - <<'PY'
import importlib.util
print(importlib.util.find_spec('genesis'))
PY
```

Result: `None`; real Genesis package is not installed on the H200 target.

- Local verification command:

```bash
PYTHONPATH=src python -m pytest -p no:cacheprovider
```

Result: `6 passed`.

# Review

Result: partial.
Syntax: pass.
Hack: pass; local backend is explicitly contract-only and does not claim physics fidelity.
Scope: pass; adapter boundary and tests only.
Efficiency: pass; no global scene singleton and no per-step logging.
Hardware: pending; real Genesis H200 reset/step cannot run until the package is installed.
Verify: local reset/step boundary passed; real Genesis reset/step pending.
Findings: next blocker is installing or transferring `genesis-world` and any required dependencies onto H200 without relying on target outbound network.
