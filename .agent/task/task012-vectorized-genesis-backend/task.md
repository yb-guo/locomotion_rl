# Task 012: Vectorized Genesis Backend

## Goal

Build the first minimal `VectorizedGenesisBackend` around the official Genesis
batched/tensor API selected in task011.

This task is a backend smoke task, not a training task. It must prove that the
27DoF no-hand G1 training asset profile can drive a batched Genesis scene with
CUDA tensors, selected reset, and fixed observation/action shapes.

## Scope

- Use branch `codex/task012-vectorized-genesis-backend`.
- Use worktree `../_worktrees/h200-locomotion-lab-task012-vectorized-genesis-backend`.
- Start from the task011 decision/profile evidence.
- Load `configs/robots/unitree_g1_27dof_nohand_genesis.yaml`.
- Add a minimal backend boundary that can:
  - build a Genesis batched scene for the 27DoF no-hand asset;
  - reset all envs or selected env ids;
  - step normalized position-delta actions shaped `[n_envs, 27]`;
  - return observations shaped `[n_envs, 90]`;
  - keep action/state tensors on CUDA during H200 runs.
- Keep 29DoF SONIC policy-contract code strict and unchanged.
- Keep `GenesisG1SceneBackend` unchanged.

## Non-Goals

- No PPO.
- No LocoFormer.
- No SONIC, ONNX, planner, GIF/video, or render in backend benchmark loops.
- No asset, dataset, checkpoint, or upstream repo downloads.
- No writes/deletes under `/mnt/workspace` or `/mnt/workspace1`.
- No migration of old single-env tools into the new backend.

## H200 Protocol

Remote commands must use:

```bash
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'
```

All remote code, outputs, and intermediate files must stay under:

```text
/root/agent_workspace/project
```

Default benchmark GPU:

```text
CUDA_VISIBLE_DEVICES=1
physical_gpu=1
logical_cuda_device=cuda:0
```

## Acceptance

- Local tests validate backend shape/device contracts without importing Genesis
  at module import time.
- H200 smoke validates `n_envs=16,256,1024` unless an upstream stop rule blocks.
- H200 output records:
  - `status=ok|blocked|failed`
  - `n_envs`
  - `action_shape`
  - `observation_shape`
  - `action_device`
  - `qpos_device`
  - `dofs_pos_device`
  - `dofs_vel_device`
  - `root_pos_device`
  - `root_quat_device`
  - `root_vel_device`
  - `tensor_device_ok`
  - `selected_reset_changes_only_target_envs`
  - throughput excluding build time
- Read-only review finds no blocking evidence or implementation findings.

# Route

1. Add the task012 document and keep the task state pending until verification
   and review evidence are present.
2. Inspect existing Genesis adapter/test boundaries and the 27DoF no-hand
   profile loader.
3. Implement a minimal backend module with dependency injection so local tests
   can run without Genesis installed.
4. Add focused tests for profile loading, action/observation shapes, selected
   reset semantics, and no module-import Genesis dependency.
5. Run local verification.
6. Prepare H200 smoke commands; run only backend smoke/profile loops, not PPO.
7. Record Route / Log / Review before marking this task passed.

# Log

- 2026-05-09 Created task012 branch/worktree from task011 commit `c23068e`
  because the task012 backend depends on the task011 27DoF training asset
  profile. PR creation for task011 through the GitHub connector was blocked by
  GitHub API 403, so the PR must be opened manually from the pushed branch URL.
- Added `h200_locomotion_lab.envs.vectorized_genesis_backend` as a new minimal
  backend boundary. It loads the 27DoF no-hand training profile, builds an
  official Genesis batched scene, maps 27 actuator joints, supports all-env and
  selected-env reset, applies `[n_envs, 27]` normalized position-delta actions,
  and returns `[n_envs, 90]` observations. The module does not import Genesis or
  Torch at module import time.
- Added `h200_locomotion_lab.tools.vectorized_genesis_backend_smoke` for H200
  backend-only smoke/profile loops. It emits key/value metrics and excludes
  PPO, SONIC, ONNX, planner, render, GIF/video, and rewards.
- Added local fake-Genesis tests for build/reset/step shape contracts, selected
  reset target-only behavior, device-report plumbing, env-id validation, and
  the no-import-at-module-import rule.
- Local focused verification:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_vectorized_genesis_backend.py tests/test_g1_27dof_nohand_profile.py -q -p no:cacheprovider`
  -> 24 passed.
- Local full verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider`
  -> 154 passed.
- `ruff` was not run because this local Python environment does not have
  `ruff` installed.

Prepared H200 smoke commands, to be run only after this task012 code is present
under `/root/agent_workspace/project/h200-locomotion-lab-task012-vectorized-genesis-backend`:

```bash
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/h200-locomotion-lab-task012-vectorized-genesis-backend && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src timeout 240 python -m h200_locomotion_lab.tools.vectorized_genesis_backend_smoke --n-envs 16 --backend cuda --physical-gpu 1 --logical-cuda-device cuda:0 --warmup-policy-steps 20 --measure-policy-steps 100'
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/h200-locomotion-lab-task012-vectorized-genesis-backend && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src timeout 360 python -m h200_locomotion_lab.tools.vectorized_genesis_backend_smoke --n-envs 256 --backend cuda --physical-gpu 1 --logical-cuda-device cuda:0 --warmup-policy-steps 20 --measure-policy-steps 100'
/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/h200-locomotion-lab-task012-vectorized-genesis-backend && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src timeout 480 python -m h200_locomotion_lab.tools.vectorized_genesis_backend_smoke --n-envs 1024 --backend cuda --physical-gpu 1 --logical-cuda-device cuda:0 --warmup-policy-steps 20 --measure-policy-steps 100'
```
- Copied the task012 code and the task011 27DoF profile files into
  `/root/agent_workspace/project/h200-locomotion-lab-task012-vectorized-genesis-backend`.
  No remote files were written outside `/root/agent_workspace/project`.
- H200 guarded focused verification:
  `PYTHONPATH=src python -m pytest tests/test_vectorized_genesis_backend.py tests/test_g1_27dof_nohand_profile.py -q -p no:cacheprovider`
  -> 24 passed.
- The first H200 smoke attempt exposed a Python 3.11 syntax issue in
  `format_key_value`; the local Python accepted an f-string expression with a
  raw string, but remote Python rejected it. The smoke tool was fixed to escape
  newlines before formatting.
- Review found that the first smoke version classified selected reset using
  DOF positions only. The backend itself reset root qpos and DOFs, but the
  evidence was too weak. The smoke tool now compares root qpos plus DOF
  positions for target-only selected-reset evidence, and the local fake-Genesis
  test asserts non-target root qpos rows remain unchanged.
- The `n_envs=256` and `n_envs=1024` smoke runs were rerun sequentially to
  avoid concurrent GPU contention from the first parallel attempt.

H200 backend smoke results:

| n_envs | status | build_time_s | observation_shape | env_policy_steps_per_sec | env_sim_steps_per_sec | tensor_device_ok | selected_reset_changes_only_target_envs |
| ---: | --- | ---: | --- | ---: | ---: | --- | --- |
| 16 | ok | 51.893809 | 16x90 | 590.051095 | 2360.204381 | true | true |
| 256 | ok | 28.285636 | 256x90 | 8190.880985 | 32763.523942 | true | true |
| 1024 | ok | 31.385784 | 1024x90 | 31867.962983 | 127471.851930 | true | true |

- Every H200 smoke run recorded `CUDA_VISIBLE_DEVICES=1`, `physical_gpu=1`,
  `logical_cuda_device=cuda:0`, `action_shape=<n_envs>x27`,
  `action_device=cuda:0`, `qpos_device=cuda:0`, `dofs_pos_device=cuda:0`,
  `dofs_vel_device=cuda:0`, `root_pos_device=cuda:0`,
  `root_quat_device=cuda:0`, and `root_vel_device=cuda:0`.
- Throughput is lower than task011's raw no-hand probe because this smoke now
  exercises the backend surface: 27D action clipping/target transform, policy
  reset plumbing, and 90D observation assembly on every policy step. This is a
  backend-overhead measurement, not a raw `scene.step()` measurement.
- H200 smoke output files:
  `outputs/task012/vectorized_backend/n16.txt`,
  `outputs/task012/vectorized_backend/n256.txt`,
  `outputs/task012/vectorized_backend/n1024.txt`.
- Local full verification after the root+DOF selected-reset evidence fix:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider`
  -> 154 passed.

# Review

Status: passed.

- Read-only boundary review found no blocking findings.
- The task changed only task012 documentation, a new independent
  `VectorizedGenesisBackend`, its backend-only smoke tool, and focused tests.
  `GenesisG1SceneBackend` and 29DoF SONIC policy-contract code were not
  modified.
- Boundary scan found no implementation paths for downloads, checkpoints,
  datasets, upstream repository clones, `/mnt/workspace` writes, PPO,
  LocoFormer, SONIC, ONNX, planner, render, GIF/video, or legacy tool
  migration.
- Verification evidence is complete for this backend smoke task: local focused
  tests, local full tests, H200 guarded focused tests, and H200 guarded
  `n_envs=16,256,1024` backend smoke runs all passed with CUDA tensors on
  physical GPU 1 / logical `cuda:0`.
- Non-blocking follow-up for task013: the observation surface is intentionally
  minimal for backend smoke. Base angular velocity, projected gravity, command
  velocity, reward, and termination semantics should be replaced with
  training-quality values before PPO is enabled.
