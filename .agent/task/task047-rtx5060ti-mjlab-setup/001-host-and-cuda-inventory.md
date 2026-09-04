# 001: Host and CUDA Inventory

## Route

Record the new workstation boundary before installing or training:

- OS and kernel;
- GPU model, VRAM, driver, and reported CUDA compatibility;
- available Python/environment managers;
- existing GPU processes and containers;
- a real CUDA tensor computation.

## Log

- 2026-08-18 `nvidia-smi` reports one RTX 5060 Ti with 16311 MiB VRAM,
  driver 595.84, and CUDA compatibility 13.2.
- 2026-08-18 Host OS is Ubuntu 22.04.5 with kernel 6.8.0-136-generic.
- 2026-08-18 `/usr/bin/python3` is Python 3.10.12 and has no pip; the repo
  requires Python >=3.11. `uv 0.12.5` is available.
- 2026-08-18 Disk preflight reports about 764 GiB free.
- 2026-08-18 The GPU is already visible to an existing Unitree Isaac Sim 5.1
  Docker container and an active `unitree_mujoco` graphics process.
- 2026-08-18 PyTorch `2.13.0+cu130` reports CUDA available, device
  `NVIDIA GeForce RTX 5060 Ti`, capability `(12, 0)`, and includes `sm_120` in
  its compiled architecture list. A real `1024 x 1024` CUDA matrix multiply
  completed with all finite outputs.

## Review

Status: passed.

The host satisfies the runtime boundary with evidence from an actual PyTorch
CUDA kernel, not only device enumeration. The workstation has one 16 GiB GPU,
so H200-scale `8192`-environment defaults must not be copied without a separate
throughput and memory sweep.
