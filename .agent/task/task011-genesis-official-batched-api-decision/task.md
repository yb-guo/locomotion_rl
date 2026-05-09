# Task 011: Genesis Official Batched API Decision

## Goal

Use the official Genesis batched/tensor API path to decide whether H200 +
Genesis should be the next vectorized PPO backend route.

This task is diagnostic. It must separate:

- H200 + Genesis CUDA stack health;
- official batched API behavior;
- locomotion/floating-base behavior;
- target G1 asset/backend suitability.

The probe order is fixed:

```text
Franka -> Go2 -> G1 -> route decision
```

## Scope

- Add an independent official batched API probe tool.
- Use official assets already prepared under `/root/agent_workspace/project/genesis_assets`.
- Run Franka first to validate baseline batched MJCF/tensor API behavior.
- Run Go2 second to validate official locomotion-style URDF/floating-base behavior.
- Run G1 third only if upstream probes pass.
- Real selected reset must be verified, not inferred from API signatures.
- Build time and steady-state step time must be measured separately.
- GPU/tensor device must be checked for action and state tensors.
- GPU/memory snapshots must be recorded before and after benchmark sections.
- Produce a decision report that chooses the next backend route.

## Non-Goals

- No PPO training.
- No LocoFormer training.
- No real Unitree SDK integration.
- No modification to `GenesisG1SceneBackend`.
- No migration of historical `tools/` scripts.
- No render, GIF/video, SONIC, ONNX, or planner subprocess in benchmark loops.
- No new asset, dataset, checkpoint, or upstream repo downloads.
- No environment upgrades unless explicitly requested.
- No default physics or asset simplification before a baseline run.

## Assets

Prepared official Genesis v0.4.6 asset bundle:

- Manifest:
  `/root/agent_workspace/project/genesis_assets/ASSET_MANIFEST.md`
- Franka:
  `/root/agent_workspace/project/genesis_assets/genesis-world-0.4.6/genesis/assets/xml/franka_emika_panda/panda.xml`
- Go2:
  `/root/agent_workspace/project/genesis_assets/genesis-world-0.4.6/genesis/assets/urdf/go2/urdf/go2.urdf`
- G1 existing SONIC asset:
  `/root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml`

Do not download additional assets in this task.

## Benchmark Protocol

Default loop:

- prefer physical GPU 1 by setting `CUDA_VISIBLE_DEVICES=1`
- `backend=cuda`
- `render=false`
- `sonic=false`
- `onnx=false`
- `planner=false`
- `gif=false`
- `warmup_policy_steps=20`
- `measure_policy_steps=100`
- `decimation=4`
- `n_envs=1,16,256,1024`
- `4096` is optional only if `1024` succeeds and memory/throughput are healthy.

Each run must emit parseable key/value metrics:

```text
status=ok|blocked|failed
asset_kind=franka|go2|g1
asset_variant=default|performance_mode|convexify|decimate
cuda_visible_devices=1
physical_gpu=1
logical_cuda_device=cuda:0
backend=cuda
n_envs=...
build_time_s=...
warmup_time_s=...
measure_time_s=...
policy_steps_per_sec=...
sim_steps_per_sec=...
env_policy_steps_per_sec=...
env_sim_steps_per_sec=...
includes_build_time=false
includes_reset_time=true|false
includes_state_read=true|false
includes_action_write=true|false
includes_reward=false
includes_render=false
action_device=...
qpos_device=...
dofs_pos_device=...
dofs_vel_device=...
root_pos_device=...
root_quat_device=...
tensor_device_ok=true|false
selected_reset_supported=true|false
selected_reset_changes_only_target_envs=true|false
selected_reset_time_s=...
gpu_snapshot_before=...
gpu_snapshot_after=...
blocker=...
```

Do not sample GPU utilization inside the step loop.

`CUDA_VISIBLE_DEVICES=1` means physical GPU 1 is exposed to the process as
logical `cuda:0`. Do not hard-code `cuda:1` in benchmark code.

## Diagnosis Rules

Use a reproduce -> minimize -> instrument -> decide loop for each asset.

Run default parameters first. Optimization probes must be single-variable:

- `performance_mode=True`
- `convexify=True`
- `decimate=True`

Do not combine these switches until a single-variable probe proves why it is
needed.

## Stop Rules

- If Franka fails, classify as environment/API issue and do not continue to
  Go2 or G1.
- If Go2 fails after Franka passes, classify as locomotion/floating-base/URDF
  issue and do not continue to G1.
- If G1 fails after Franka and Go2 pass, classify as target asset/backend issue.
- If `n_envs=1` build fails, stop that asset and record the exact blocker.
- If `n_envs=16` fails, record no batched support for that asset/path.
- If higher `n_envs` fails due OOM, record max usable env count.
- If a run times out, record timeout as a result and do not keep retrying
  without changing a single diagnostic variable.
- Do not report scalar-looped envs as batched support.
- If physical GPU 1 is unavailable or `CUDA_VISIBLE_DEVICES=1` does not isolate
  the process to a single visible CUDA device, stop and record the exact GPU
  isolation blocker before running benchmarks.

## Subtasks

- `001-franka-official-batched-baseline.md`
- `002-go2-official-locomotion-baseline.md`
- `003-g1-official-api-target-probe.md`
- `004-backend-route-decision.md`

## Acceptance

- Each subtask has `Route / Log / Review`.
- The probe tool is independent and does not modify `GenesisG1SceneBackend`.
- Franka, Go2, and G1 subtasks each record pass/block/fail status or are
  explicitly skipped by stop rule.
- Selected reset is actually exercised where the subtask runs.
- Tensor device checks are recorded and treated as hard requirements.
- The decision report states one route:
  - proceed to `VectorizedGenesisBackend`;
  - create a G1 training asset simplification task;
  - fix Genesis/Torch/H200 environment first;
  - switch PPO baseline to another backend.
- Read-only review finds no blocking benchmark or evidence issues.

## Result

pending

## Review

Status: pending.
