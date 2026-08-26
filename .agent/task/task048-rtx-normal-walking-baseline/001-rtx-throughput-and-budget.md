# 001: RTX Throughput and Budget

## Route

Measure `Unitree-G1-Flat` PPO throughput at increasing environment counts on
the single 16 GiB RTX 5060 Ti. Use the normal 24-step rollout and multiple
iterations so cached-kernel steady-state speed is visible. Stop increasing at
OOM, instability, or a clear throughput plateau.

## Log

- 2026-08-19 GPU preflight found no active compute process and only 15 MiB
  reported in use before probing.
- 2026-08-19 Steady-state probes with 24 rollout steps measured:
  - 1024 envs: about `39.9k` steps/s;
  - 2048 envs: about `54.6k` steps/s;
  - 4096 envs: peak `59.5k` steps/s;
  - 6144 envs: peak `56.8k` steps/s;
  - 8192 envs: peak `52.6k` steps/s.
- 2026-08-19 The 8192-env probe completed without OOM and an external
  `nvidia-smi` sample observed about 5.9 GiB used, but throughput had already
  declined. Selected 4096 envs for the long run rather than maximizing batch
  size.

## Review

Status: passed. The selected long-run point is 4096 envs, 24 rollout steps,
seed 42, with expected steady-state throughput around 53-59k steps/s.
