# Route

Task: task004-genesis-g1-baseline

Goal: Define robot asset, observation, action, and control-rate contract.

Scope:

- `configs/envs/genesis_g1.yaml`
- `src/h200_locomotion_lab/envs`
- docs if needed

Verify:

- Observation fields, action fields, control rate, and units are documented.

Environment:

- local docs/code

No Hack:

- no hidden asset path
- no unverified joint order
- no undocumented unit conversion

Hardware:

- not hardware-bound

# Log

- 2026-05-06: Added the Genesis G1 contract to `configs/envs/genesis_g1.yaml`.
- Joint order was inventoried from the existing H200 SONIC source snapshot at
  `gear_sonic/data/robots/g1/g1_29dof.xml`; no new assets were downloaded.
- Contract records:
  - 29 DoF joint order.
  - 50 Hz policy rate, 200 Hz sim rate, decimation 4.
  - 96D observation layout and units.
  - 29D normalized action mapped to joint position delta in radians.
- Local verification command:

```bash
PYTHONPATH=src python -m pytest -p no:cacheprovider
```

Result: `6 passed`.

# Review

Result: pass for local contract.
Syntax: pass.
Hack: pass; no hidden asset path, asset remains explicit external path.
Scope: pass; only config and adapter contract were changed.
Efficiency: pass; no runtime hot path added.
Hardware: not hardware-bound.
Verify: `PYTHONPATH=src python -m pytest -p no:cacheprovider` -> `6 passed`.
Findings: H200 real Genesis package is not installed yet; this subtask only freezes the contract.
