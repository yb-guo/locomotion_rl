# Route

Task: task003-h200-simulator-smoke

Goal: Verify Genesis can run a headless minimal scene on H200.

Scope:

- Genesis install/import notes
- one reset/step smoke script or command

Verify:

- Genesis imports.
- CUDA device is visible if used.
- A small headless scene steps successfully.

Environment:

- Linux H200 target

No Hack:

- no GUI dependency
- no unbounded asset download
- no treating CPU-only fallback as H200 success without labeling it

Hardware:

- record driver, CUDA, GPU name, and memory

# Log

- 2026-05-06: Installed `genesis-world==0.4.6` on the H200 target from a local
  transferred wheelhouse because target outbound network is unreliable.
- H200 base environment details for this smoke:
  - Host: `nb-0p4ivpp1kj-0`
  - GPU: `NVIDIA H200`
  - Python: `Python 3.11.11`
  - Torch: `2.5.1+cu124`
  - Genesis: `0.4.6`
- Known environment caveat: this was installed into the existing base conda env,
  which now has dependency conflicts with the pre-existing GR00T stack and a
  Genesis warning that `torch<2.8.0` is unsupported. Use a separate env for
  sustained Genesis work.
- Minimal CUDA plane smoke passed:

```text
Command script: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/genesis_plane_cuda_smoke.py
Log: /root/h200-locomotion-lab-runs/task004-genesis-g1-baseline/logs/genesis_plane_cuda_smoke.log
GENESIS_VERSION 0.4.6
Running on [NVIDIA H200] with backend gs.cuda
PLANE_CUDA_SMOKE_OK steps=20 elapsed_s=3.164
PLANE_CUDA_EXIT_STATUS=0
```

# Review

Result: pass.
Syntax: pass.
Hack: pass; headless scene, no GUI dependency, no simulator assets downloaded.
Scope: pass; import/build/step only.
Efficiency: pass for minimal smoke.
Hardware: pass; Genesis selected `gs.cuda` on `NVIDIA H200`.
Verify: `PLANE_CUDA_SMOKE_OK steps=20` with exit status `0`.
Findings: use an isolated environment before any longer Genesis runs because
the base env dependency graph is now mixed with GR00T/SONIC packages.
