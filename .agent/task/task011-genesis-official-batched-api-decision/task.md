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

passed

Decision: proceed to `VectorizedGenesisBackend`.

Evidence:

- Independent official batched API probe added:
  `h200_locomotion_lab.tools.genesis_official_batched_api_probe`.
- The probe does not modify or use `GenesisG1SceneBackend`.
- Local verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider`
  -> 125 passed; focused probe tests cover 16 cases.
- H200 guarded focused verification:
  `PYTHONPATH=src python -m pytest tests/test_genesis_official_batched_api_probe.py -q -p no:cacheprovider`
  -> 16 passed.
- Remote evidence is saved under
  `/root/agent_workspace/project/h200-locomotion-lab-task011-genesis-official-batched-api-decision/outputs/task011`.
- Benchmark GPU isolation:
  `CUDA_VISIBLE_DEVICES=1`, `physical_gpu=1`,
  `logical_cuda_device=cuda:0`.
- Franka official MJCF default path passed through `n_envs=1024`.
- Go2 official URDF/floating-base default path passed through `n_envs=1024`.
- G1 target default MJCF path passed through `n_envs=1024`; optional
  `n_envs=4096` also passed.
- Batched runs verified CUDA tensor devices, selected reset target-only
  behavior, and no render/GIF/SONIC/ONNX/planner/PPO benchmark-loop work.
- Follow-up G1 single-variable diagnostics at `n_envs=1024` found
  `performance_mode=True` is not accepted by Genesis 0.4.6 `MJCF`;
  `convexify=True` and `decimate=True` did not materially improve steady-state
  throughput versus the default baseline.
- Follow-up component profiling found raw `scene.step()` is the bottleneck:
  G1 29DoF needed 4.380875s for 400 raw scene steps at `n_envs=1024`, while
  Go2 needed 0.522974s. Action write and state read were millisecond-scale.
- Existing simpler G1 assets were inventoried without downloads. The directly
  runnable faster candidate is `g1_27dof_nohand.xml`, which reached
  45827.527990 env policy steps/s at `n_envs=1024`; `g1_27dof_fakehand.xml`
  reached 44164.266383. Several 12DoF/23DoF candidates are present but blocked
  by existing mesh/importer issues.
- A standalone `g1_27dof_nohand.xml` training asset profile was added for the
  `VectorizedGenesisBackend` route:
  `configs/robots/unitree_g1_27dof_nohand_genesis.yaml`.
  It uses a separate loader,
  `h200_locomotion_lab.robots.g1_27dof_nohand`, so the 29DoF SONIC robot
  profile remains strict. The profile records the H200 guarded benchmark
  evidence, component-profile evidence, 27D action size, and 90D observation
  contract.
- Focused profile verification:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q -p no:cacheprovider`
  -> 27 passed.

## Review

Status: passed.

Initial read-only review found two blocking evidence gaps: floating-base
selected reset only covered DOFs, and root velocity device was not measured.
The probe was fixed to emit `root_vel_device` and to exercise selected root
qpos plus selected joint reset for Go2/G1. A second read-only review found no
blocking or important findings.

Training-profile re-review addendum:

- A read-only reviewer initially inspected the wrong root
  `D:\guoyubo.9\Documents\New project 2\h200-locomotion-lab`, so those
  findings were invalid for this task worktree.
- A corrected read-only review against
  `D:\guoyubo.9\Documents\New project 2\_worktrees\h200-locomotion-lab-task011-genesis-official-batched-api-decision`
  found one blocking issue: the new 27DoF loader validated uniqueness but not
  canonical 29DoF-derived actuator order. The loader was fixed to enforce the
  canonical `g1_27dof_nohand.xml` actuator order and the
  `joint_order.order` / `joint_order.derived_from` schema fields.
- Focused verification after the fix:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q -p no:cacheprovider`
  -> 27 passed.
- Full local verification after the profile fix:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider`
  -> 146 passed.
- Corrected read-only re-review after the fix found no blocking findings.
