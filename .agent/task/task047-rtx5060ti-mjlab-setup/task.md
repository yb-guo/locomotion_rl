# Task 047: RTX 5060 Ti MJLab Setup

## Route

Bring the current Unitree MJLab / LocoFormer-style training path up on the new
single-GPU workstation without carrying over H200-only assumptions.

The target machine is an Ubuntu 22.04 workstation with one RTX 5060 Ti 16 GiB.
This task separates four claims:

1. the host and CUDA runtime can execute PyTorch kernels on the GPU;
2. the local project has a reproducible Python 3.11 environment;
3. the official Unitree MJLab baseline can import and step on the GPU;
4. the project-specific Task044/Task046 consumer can be constructed and run a
   minimal PPO smoke.

Do not claim the old H200 policy is restored unless its exact checkpoint and
matching patched MJLab source are present. Do not download checkpoints,
datasets, reference motions, or optional simulator assets.

## Planned Slices

1. `001-host-and-cuda-inventory.md`
   - Record GPU, driver, OS, Python, disk, and existing simulator/container
     state.
   - Execute a real CUDA tensor kernel, not only `nvidia-smi`.

2. `002-python-and-upstream-mjlab.md`
   - Create an isolated Python 3.11 environment.
   - Install the local project and the official Unitree MJLab dependency stack.
   - Record the upstream revision and exact resolved package versions.

3. `003-project-adapter-registration.md`
   - Reconstruct only the minimum project-specific MJLab registration needed
     for the current retry-context/hidden-fault consumer.
   - Keep upstream source modifications explicit and repeatable.

4. `004-rtx-gpu-smoke.md`
   - Run local tests and `inspect_agent`.
   - Run a small official Unitree G1 GPU env/runner smoke.
   - Run the smallest project-specific Task044/Task046 PPO consumer smoke if
     its dependencies and source registration are available.

## Acceptance Criteria

- A Python 3.11 environment has a working `python` entry point.
- PyTorch reports CUDA available and executes a finite CUDA computation on the
  RTX 5060 Ti.
- MuJoCo/MJLab import versions and the upstream source revision are recorded.
- `python -m pytest` and `inspect_agent` have verification evidence.
- At least one real GPU simulator step or runner smoke succeeds.
- Missing old H200 checkpoint/source state is reported as a migration boundary,
  not silently treated as success.
- VRAM-safe RTX defaults are recorded separately from the old 8192-env H200
  settings.

## Log

- 2026-08-18 Opened after the user requested setup on the new GPU.
- 2026-08-18 Host inventory found Ubuntu 22.04.5, NVIDIA driver 595.84,
  system-reported CUDA 13.2, and one RTX 5060 Ti with 16 GiB VRAM.
- 2026-08-18 The host initially had Python 3.10 only on `python3`, no `python`
  command, and no host pip. `uv 0.12.5` is available and will provide the
  isolated Python 3.11 environment.
- 2026-08-18 An existing `unitree-rl-full:isaacsim5.1` container and Unitree
  MuJoCo process were found, but the current project uses the distinct
  Unitree MJLab/MuJoCo path. The old H200 paths under `/mnt/workspace/users/...`
  and their checkpoints are absent on this host.
- 2026-08-18 Created the Python 3.11 environment and pinned the verified RTX
  stack: PyTorch `2.13.0+cu130`, MJLab `1.2.0`, MuJoCo/MuJoCo-Warp `3.5.0`,
  Warp `1.12.0`, RSL-RL `5.0.1`, and SciPy `1.17.1`. The reproducible setup
  entry point is `scripts/setup_rtx_mjlab.sh`.
- 2026-08-18 CUDA tensor preflight, official 29-action G1 PPO, and rebuilt
  Task028 31-action G1-gripper PPO all passed on the RTX 5060 Ti with 32 envs
  and a one-iteration smoke budget.
- 2026-08-18 Complete local verification passed with `713` tests. The setup
  added portability handling for inaccessible legacy absolute paths, POSIX
  parsing of Windows asset paths, and legacy true-TXL debug snapshots.
- 2026-08-18 Task044/046 is not claimed restored. The tracked history lacks a
  complete reproduction of the Task029-031 changes that lived in the old
  cumulatively patched Unitree checkout, and the matching policy checkpoints
  are absent. `scripts/check_task044_migration.sh` records this boundary with
  a failing exit status.

## Review

Status: partially complete; do not mark the full current algorithm passed.

The new GPU is fully set up for the official Unitree MJLab baseline and the
reconstructable Task028 31-action morphology layer. Both performed a real GPU
PPO update and saved a checkpoint. The remaining Task044/046 consumer cannot be
verified from the material present on this workstation: Task029-031 upstream
environment source and the old matching checkpoints must be migrated or
reconstructed first. This is now an explicit, executable audit failure rather
than an implicit setup assumption.
