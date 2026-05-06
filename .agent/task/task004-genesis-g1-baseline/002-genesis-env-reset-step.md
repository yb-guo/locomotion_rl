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

- 2026-05-06: Installed `genesis-world==0.4.6` plus its Linux/Python 3.11
  dependency wheelhouse on the H200 target under the base conda environment.
  The target has `torch==2.5.1+cu124`, so Genesis warns that `torch<2.8.0`
  is unsupported. `pip check` also reports dependency conflicts with the
  pre-existing GR00T environment and a `pygltflib` pin mismatch. This install is
  acceptable for smoke testing, but repeated SONIC/Genesis work should use a
  separate environment.
- H200 minimal Genesis CUDA plane smoke passed:

```text
Log: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/genesis_plane_cuda_smoke.log
GENESIS_VERSION 0.4.6
Running on [NVIDIA H200] with backend gs.cuda
PLANE_CUDA_SMOKE_OK steps=20 elapsed_s=3.164
PLANE_CUDA_EXIT_STATUS=0
```

- H200 SONIC G1 MJCF attempt using the original contract inventory path failed
  because the mesh files below `gear_sonic/data/robots/g1/meshes` are still Git
  LFS pointer text files, not real STL meshes:

```text
Log: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/genesis_sonic_g1_mjcf_cuda_smoke.log
Asset: /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml
ValueError: decoder failed for mesh file .../left_hip_roll_link.STL
```

- H200 SONIC G1 MJCF smoke passed when using the already materialized SONIC
  `model_data/g1` asset path:

```text
Log: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/genesis_sonic_g1_model_data_mjcf_cuda_smoke.log
Asset: /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml
GENESIS_VERSION 0.4.6
Running on [NVIDIA H200] with backend gs.cuda
ROBOT_N_DOFS 49
ROBOT_N_LINKS 44
SONIC_G1_MODEL_DATA_MJCF_CUDA_SMOKE_OK steps=20 elapsed_s=3.861
SONIC_G1_MODEL_DATA_MJCF_CUDA_EXIT_STATUS=0
```

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
Hardware: pass for raw Genesis CUDA import/build/step smoke on H200.
Verify: local reset/step boundary passed; raw Genesis H200 plane and SONIC G1
MJCF build/step smoke passed.
Findings: the repo environment wrapper still has only a contract-only backend.
The passing SONIC asset smoke used `g1_29dof_with_hand.xml`, which Genesis
reports as `49` DoF, so the next implementation step is selecting or exporting a
true 29DoF Genesis-compatible G1 asset before PPO training.
