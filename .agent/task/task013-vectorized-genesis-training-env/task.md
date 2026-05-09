# Task 013: Vectorized Genesis Training Env

## Goal

Build a training-efficient velocity-tracking env on top of the task012
`VectorizedGenesisBackend`.

This task is still not PPO. It proves the env contract needed before PPO:
batched observations, rewards, done signals, command sampling, and selected
reset semantics all on CUDA.

## Scope

- Use branch `codex/task013-vectorized-genesis-training-env`.
- Use worktree `../_worktrees/h200-locomotion-lab-task013-vectorized-genesis-training-env`.
- Add `G1VelocityTrackingVectorizedEnv` for the 27DoF no-hand G1 Genesis asset.
- Keep env/policy/trainer separated:
  - env consumes actions shaped `[n_envs, 27]`;
  - env returns observations shaped `[n_envs, 90]`;
  - env returns reward, terminated, truncated, and done shaped `[n_envs]`;
  - no policy network, rollout buffer, optimizer, PPO, or LocoFormer.
- Use command space `vx + yaw_rate`; keep `vy=0` while preserving 3D command
  observation segment `[vx, vy, yaw_rate]`.
- Implement height + tilt + timeout done logic.
- Add small backend API changes only where needed for training efficiency:
  read batched state once and step physics without building placeholder backend
  observations.
- Keep 29DoF SONIC policy-contract code strict and unchanged.
- Keep `GenesisG1SceneBackend` unchanged.

## Non-Goals

- No PPO.
- No LocoFormer.
- No SONIC, ONNX, planner, GIF/video, or render in env benchmark loops.
- No asset, dataset, checkpoint, or upstream repo downloads.
- No writes/deletes under `/mnt/workspace` or `/mnt/workspace1`.
- No contact-based reward or contact-based termination in this task.

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

- Local tests validate env contract without importing Genesis at module import
  time.
- Local full pytest passes.
- H200 smoke validates `n_envs=16,256,1024` unless an upstream stop rule blocks.
- H200 output records:
  - `status=ok|blocked|failed`
  - `n_envs`
  - `action_shape`
  - `observation_shape`
  - `reward_shape`
  - `terminated_shape`
  - `truncated_shape`
  - `done_shape`
  - `command_shape`
  - tensor device fields for obs/reward/done/commands/state
  - `tensor_device_ok`
  - `selected_reset_changes_only_target_envs`
  - `done_reset_resets_only_done_envs`
  - reward component means
  - throughput excluding build time
- Target H200 performance for `n_envs=1024`: at least
  `20000 env_policy_steps_per_sec` if Genesis upstream behavior remains
  comparable to task012.
- Read-only review finds no blocking evidence or implementation findings.

# Route

1. Create task013 branch/worktree from merged `master`.
2. Add task013 document with stop rules and H200 protocol.
3. Add efficient backend state/physics helpers without changing
   `GenesisG1SceneBackend`.
4. Add `G1VelocityTrackingVectorizedEnv` with vx+yaw command sampling, real
   observation assembly, reward components, height/tilt/timeout done, and
   selected done reset.
5. Add backend/env smoke tool that records shape, device, reward, done, reset,
   and throughput metrics.
6. Add local fake-Genesis tests.
7. Run local verification.
8. Run H200 guarded smoke on physical GPU 1.
9. Record Log / Review before marking passed.

# Log

- 2026-05-09 Created task013 branch/worktree from merged `master` commit
  `d7d842c`.
- Design decisions from grill:
  - task013 is env semantics/performance only, not PPO;
  - env and policy remain separated;
  - small `VectorizedGenesisBackend` API changes are allowed for efficiency;
  - command space is `vx + yaw_rate`, with `vy=0`;
  - done logic is height + tilt + timeout;
  - reward component metrics are recorded only in smoke summaries.
- Added `h200_locomotion_lab.envs.g1_velocity_tracking_env` with
  `G1VelocityTrackingVectorizedEnv`. It keeps commands, episode lengths,
  actions, observations, rewards, done tensors, and reward components in the
  backend tensor domain.
- Added backend helpers:
  - `VectorizedGenesisState` for one-place batched state reads;
  - `step_physics()` to advance Genesis without building placeholder backend
    observations;
  - tensor env-id reset support for done-env reset;
  - root angular velocity read with zero fallback when Genesis does not expose
    an angular-velocity getter.
- Added `h200_locomotion_lab.tools.g1_velocity_tracking_env_smoke` for
  backend/env-only H200 smoke. It records shape, device, reward component,
  done/reset, and throughput metrics. It does not run PPO, LocoFormer, SONIC,
  ONNX, planner, render, or GIF/video.
- Added fake-Genesis tests covering no Genesis import at module import time,
  reset/step shapes, reward/done components, timeout selected reset,
  height-based termination reset, backend state/physics helpers, and device
  report plumbing.
- Local focused verification:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_velocity_tracking_env.py tests/test_vectorized_genesis_backend.py -q -p no:cacheprovider`
  -> 15 passed.
- Local full verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider`
  -> 172 passed.
- `ruff` was not run because this local Python environment does not have
  `ruff` installed.
- Remote task013 workspace was created under
  `/root/agent_workspace/project/h200-locomotion-lab-task013-vectorized-genesis-training-env`
  by copying the prior project workspace and then copying task013 files into
  that project directory. No remote files were written outside
  `/root/agent_workspace/project`.
- H200 guarded focused verification:
  `PYTHONPATH=src python -m pytest tests/test_g1_velocity_tracking_env.py tests/test_vectorized_genesis_backend.py -q -p no:cacheprovider`
  -> 15 passed.

H200 env smoke results:

| n_envs | status | build_time_s | observation_shape | reward_shape | env_policy_steps_per_sec | env_sim_steps_per_sec | tensor_device_ok | selected_reset_changes_only_target_envs | done_reset_resets_only_done_envs | reward_mean |
| ---: | --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | ---: |
| 16 | ok | 30.578780 | 16x90 | 16 | 1256.432819 | 5025.731277 | true | true | true | 1.305187 |
| 256 | ok | 31.815598 | 256x90 | 256 | 18031.746062 | 72126.984250 | true | true | true | 1.461088 |
| 1024 | ok | 31.363309 | 1024x90 | 1024 | 67665.142756 | 270660.571026 | true | true | true | 1.453202 |

- Every H200 smoke run recorded `CUDA_VISIBLE_DEVICES=1`, `physical_gpu=1`,
  `logical_cuda_device=cuda:0`, action/observation/reward/done/command/state
  tensors on `cuda:0`, `timeout_count=0`, `fallen_count=0`,
  `action_rate_penalty_mean=0.000000`, and small positive
  `joint_deviation_penalty_mean`.
- `n_envs=1024` exceeded the task target of `20000 env_policy_steps_per_sec`.
  The env loop is faster than task012 backend smoke because it uses
  `step_physics()` plus one explicit env state read instead of building the
  placeholder backend observation through `backend.step()`.
- H200 smoke output files:
  `outputs/task013/g1_velocity_tracking_env/n16.txt`,
  `outputs/task013/g1_velocity_tracking_env/n256.txt`,
  `outputs/task013/g1_velocity_tracking_env/n1024.txt`.

# Review

Status: passed.

- Read-only boundary review found no blocking findings.
- The task changed only task013 documentation, the independent
  velocity-tracking env, its backend/env-only smoke tool, tests, and small
  task013-required helper additions to `VectorizedGenesisBackend`.
  `GenesisG1SceneBackend` and 29DoF SONIC policy-contract code were not
  modified.
- Boundary scan found no implementation paths for downloads, checkpoints,
  datasets, upstream repository clones, `/mnt/workspace` writes, PPO,
  LocoFormer, SONIC, ONNX, planner, render, GIF/video, or legacy tool
  migration. Matches were limited to task document stop-rule text.
- Verification evidence is complete for this env smoke task: local focused
  tests, local full tests, H200 guarded focused tests, and H200 guarded
  `n_envs=16,256,1024` env smoke runs all passed with CUDA tensors on physical
  GPU 1 / logical `cuda:0`.
- Non-blocking follow-up for task014: PPO smoke can now bind to this env
  contract, but reward weights are still first-pass smoke-stable values rather
  than tuned locomotion values.
