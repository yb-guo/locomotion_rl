# 002: Python and Upstream MJLab

## Route

Use `uv` to create an isolated Python 3.11 environment for this repo, then
install the project and official Unitree MJLab stack. Pin or record the exact
resolved versions and upstream git revision so the setup is repeatable.

The official setup contract is Ubuntu 22.04, Python 3.11, NVIDIA GPU, and
driver 550 or later. The upstream package currently declares `mjlab==1.2.0`
and `mujoco-warp==3.5.0`.

Do not fetch checkpoints, reference motions, datasets, or optional assets.

## Log

- 2026-08-18 Created `/home/admin1/workspace/proj/locomotion_rl/.venv` with
  `uv` and CPython `3.11.16`.
- 2026-08-18 Created a shallow sparse official checkout at
  `.external/unitree_rl_mjlab`, revision
  `1425b15f73bd4095f0df53709d7c389c3eb9e790` (`Fix the warnings during
  rough-terrain training.`). No checkpoint, reference motion, dataset, or
  optional simulator asset was fetched.
- 2026-08-18 The upstream metadata only pins `mjlab==1.2.0` and
  `mujoco-warp==3.5.0`. Current unconstrained resolution exposed three missing
  compatibility constraints: MuJoCo 3.11 removed
  `mjtEnableBit.mjENBL_MULTICCD`, Warp 1.16 removed `warp.context`, and the
  task registry imports undeclared SciPy.
- 2026-08-18 Verified stack:
  - PyTorch `2.13.0+cu130`;
  - torchvision `0.28.0`;
  - MJLab `1.2.0`;
  - MuJoCo `3.5.0`;
  - MuJoCo Warp `3.5.0`;
  - Warp `1.12.0`;
  - RSL-RL `5.0.1`;
  - SciPy `1.17.1`;
  - NumPy `2.4.6`.
- 2026-08-18 Added the verified constraint set at
  `configs/requirements/rtx5060ti-mjlab-constraints.txt` and the idempotent
  installer `scripts/setup_rtx_mjlab.sh`. A clean rerun resolved 135 packages,
  `uv pip check` reported all 149 installed packages compatible, and printed
  the expected versions.
- 2026-08-18 Launch boundary: Unitree commands must run from the upstream
  checkout root with this repo's absolute `src` path prepended to `PYTHONPATH`.
  Both repositories expose a top-level Python package named `src`, so running
  the Unitree entry point from this repo root alone selects the wrong package.

## Review

Status: passed.

The Python environment, official source revision, compatibility pins, import
boundary, and reproducible installer are all recorded and rerun successfully.
