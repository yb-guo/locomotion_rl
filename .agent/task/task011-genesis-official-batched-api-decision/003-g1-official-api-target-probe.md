# Route

Task: task011-genesis-official-batched-api-decision

Goal: determine whether the current SONIC G1 29DoF asset can use the official
Genesis batched/tensor API path after Franka and Go2 pass.

Scope:

- Use the existing SONIC G1 29DoF asset path.
- Start with the original/default asset settings.
- Build with `n_envs=1,16,256,1024` subject to stop rules.
- Verify tensor action write and tensor root/DOF state reads.
- Verify selected root + joint reset on target envs only.
- If default G1 is blocked or very slow, run only single-variable probes:
  `performance_mode=True`, `convexify=True`, or `decimate=True`.
- Record whether a separate training asset profile is needed.

Environment:

- interactive remote sessions must enter `/root/agent_workspace/safe_agent/agent_shell.sh`
- non-interactive remote commands must run through `/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'`
- benchmark commands must prefer physical GPU 1 with `CUDA_VISIBLE_DEVICES=1`
- record that physical GPU 1 maps to logical `cuda:0` inside the process
- record `nvidia-smi` GPU/memory snapshot before and after the benchmark section
- code, generated files, and intermediate outputs must stay under `/root/agent_workspace/project`
- datasets must be accessed only through `/root/agent_workspace/datasets`
- dataset links must be created with `/root/agent_workspace/safe_agent/link_dataset.sh`
- do not write to or delete from `/mnt/workspace` or `/mnt/workspace1`
- do not manually clear `LD_PRELOAD`

Verify:

- Franka and Go2 subtasks passed or this subtask is skipped by stop rule.
- default G1 run is attempted before optimization probes.
- `status=ok|blocked|failed` for each attempted variant and `n_envs`.
- `cuda_visible_devices=1`, `physical_gpu=1`, and `logical_cuda_device=cuda:0`
  are recorded or a GPU isolation blocker is recorded.
- `tensor_device_ok=true` is required for pass.
- Selected root + joint reset changes only target envs.
- G1 blocker is classified as asset, tensor IO, selected reset, timeout, OOM,
  or environment.

No Hack:

- Do not alter the official SONIC validation asset in place.
- Do not make convexify/decimate the default without documenting it as a
  separate training asset decision.
- Do not use scalar wrapper loops as batched support.

Hardware: H200/Linux target via `myserver`.

# Log

- 2026-05-09 H200 guarded G1 runs started only after Franka and Go2 both
  passed through `n_envs=1024`, satisfying the upstream stop rules.
- Read-only review found that the first probe version recorded selected DOF
  reset but not selected floating-base root reset, and did not emit root
  velocity device evidence. The probe was fixed to read `get_vel()` and to
  exercise selected root qpos reset plus selected joint reset. An attempted
  `set_pos`/`set_quat` root-reset path exposed a Genesis blocker for this G1
  asset (`batch_fixed_verts=True` would be needed for selected fixed-link pose
  writes), so the final probe uses the official selected `set_qpos` root path.
  The G1 matrix below was rerun after that fix.
- H200 guarded benchmark command shape:
  `/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/h200-locomotion-lab-task011-genesis-official-batched-api-decision && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src timeout ... python -m h200_locomotion_lab.tools.genesis_official_batched_api_probe --asset-kind g1 --asset /root/h200-locomotion-lab-runs/task002-sonic-mujoco-smoke/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml --n-envs ... --warmup-policy-steps 20 --measure-policy-steps 100 --decimation 4'`
- Default G1 output files:
  `outputs/task011/g1/default_n1.txt`,
  `outputs/task011/g1/default_n16.txt`,
  `outputs/task011/g1/default_n256.txt`,
  `outputs/task011/g1/default_n1024.txt`.
  Optional capacity evidence was also saved in
  `outputs/task011/g1/default_n4096.txt` because `n_envs=1024` succeeded with
  healthy memory.

| n_envs | status | build_time_s | measure_time_s | policy_steps_per_sec | env_policy_steps_per_sec | env_sim_steps_per_sec | tensor_device_ok | selected_reset_changes_only_target_envs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | ok | 63.829883 | 1.386982 | 72.098978 | 72.098978 | 288.395914 | true | false (not applicable for one env) |
| 16 | ok | 65.852252 | 2.804467 | 35.657400 | 570.518403 | 2282.073611 | true | true |
| 256 | ok | 66.439508 | 4.214304 | 23.728713 | 6074.550575 | 24298.202302 | true | true |
| 1024 | ok | 60.015459 | 4.598909 | 21.744286 | 22266.149334 | 89064.597337 | true | true |
| 4096 optional | ok | 28.308660 | 5.224667 | 19.139974 | 78397.335490 | 313589.341962 | true | true |

- Every default G1 run emitted:
  `cuda_visible_devices=1`, `physical_gpu=1`,
  `logical_cuda_device=cuda:0`, `backend=cuda`,
  `includes_build_time=false`, `includes_state_read=true`,
  `includes_action_write=true`, `includes_reward=false`,
  `includes_render=false`, and all recorded root/DOF tensor devices as
  `cuda:0`, including `root_vel_device=cuda:0`.
- Selected root qpos plus joint reset was actually exercised for `n_envs>=16`
  and changed only the target env. The `n_envs=1` run records
  `selected_reset_time_s=not_applicable_n_envs_lt_2`.
- G1 default asset was attempted before any optimization probe. Because the
  default asset passed through `n_envs=1024` and optional `4096` with healthy
  memory, optimization probes were not needed for the route decision. After
  the route decision, the user requested deeper performance diagnosis, so
  single-variable G1 probes were run at `n_envs=1024` with the same
  `warmup_policy_steps=20`, `measure_policy_steps=100`, and `decimation=4`.
  Diagnostic outputs are saved under `outputs/task011/g1_diagnostics/`.

| variant | status | build_time_s | measure_time_s | env_policy_steps_per_sec | env_sim_steps_per_sec | diagnosis |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| default baseline | ok | 60.015459 | 4.598909 | 22266.149334 | 89064.597337 | baseline |
| performance_mode | blocked | unavailable | unavailable | unavailable | unavailable | Genesis 0.4.6 `MJCF` rejected `performance_mode` as an unrecognized morph attribute |
| convexify | ok | 59.077321 | 4.552997 | 22490.678338 | 89962.713353 | no meaningful speedup over default |
| decimate | ok | 60.631502 | 4.567625 | 22418.654409 | 89674.617635 | no meaningful speedup over default |

- Performance diagnosis:
  `convexify=True` and `decimate=True` each changed throughput by roughly
  1 percent relative to the fixed default baseline. Therefore the observed
  G1 slowdown versus Go2 is unlikely to be solved by these Genesis morph
  switches alone. The likely causes remain G1 humanoid complexity: more DOFs,
  more links and contacts, heavier collision/constraint solve work, and the
  current target asset's mass/geometry warnings.
- Follow-up component profile:
  added `h200_locomotion_lab.tools.genesis_official_component_profile` to split
  action write, state read, raw `scene.step()`, and combined policy-loop costs.
  The component profile reuses the same official Genesis scene path and writes
  outputs under `outputs/task011/component_profile/`.

| asset | n_envs | action_write_time_s | state_read_time_s | scene_step_time_s for 400 steps | scene_steps_per_sec | combined_env_policy_steps_per_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Go2 official URDF | 1024 | 0.009735 | 0.024153 | 0.522974 | 764.856915 | 160470.049206 |
| G1 29DoF target XML | 1024 | 0.016692 | 0.021145 | 4.380875 | 91.305966 | 27803.650774 |
| G1 27DoF no-hand XML | 1024 | 0.018051 | 0.024633 | 2.134225 | 187.421643 | 29547.329979 |

- Component profile diagnosis:
  action write and state read are millisecond-scale for 100 policy iterations.
  The dominant difference is raw physics stepping: G1 29DoF reaches only
  91.3 scene steps/s while Go2 reaches 764.9 scene steps/s. The 27DoF no-hand
  asset roughly doubles raw G1 scene-step throughput to 187.4 scene steps/s.
- Existing simpler G1 asset inventory under the prepared SONIC robot directory
  includes `g1_12dof.urdf`, `g1_12dof_sausage.urdf`, `g1_23dof.xml`,
  `g1_27dof_nohand.xml`, `g1_27dof_fakehand.xml`, and many 29DoF variants.
  No assets were downloaded.

| simpler asset | status at n_envs=1024 | env_policy_steps_per_sec | blocker / note |
| --- | --- | ---: | --- |
| `g1_27dof_nohand.xml` | ok | 45827.527990 | best current direct training candidate; about 2.1x default 29DoF throughput |
| `g1_27dof_fakehand.xml` | ok | 44164.266383 | usable, slightly slower than no-hand |
| `g1_23dof.xml` | blocked | unavailable | STL decoder failed on rubber hand meshes |
| `g1_29dof_anneal_23dof.xml` | blocked | unavailable | rubber hand geometry has fewer than 4 vertices |
| `g1_12dof.urdf` | blocked | unavailable | Genesis URDF importer reported no mesh present |
| `g1_12dof_sausage.urdf` | blocked | unavailable | Genesis URDF importer reported no mesh present |

- Simpler-model diagnosis:
  There is a prepared, directly runnable simpler G1 candidate:
  `g1_27dof_nohand.xml`. It materially improves throughput, but it is still
  far slower than Go2, so the next task should treat it as a training asset
  candidate rather than assume it solves all G1 physics cost.
- Training-asset profile addendum:
  `g1_27dof_nohand.xml` is now recorded as a standalone
  `VectorizedGenesisBackend` training profile, not as a replacement for the
  strict 29DoF SONIC policy profile. The profile is
  `configs/robots/unitree_g1_27dof_nohand_genesis.yaml`, with an isolated
  loader in `h200_locomotion_lab.robots.g1_27dof_nohand`. It records
  `format=mjcf`, `genesis_morph=MJCF`, 27 actuator joints, `sim_dt_s=0.005`,
  `decimation=4`, `policy_rate_hz=50`, `action_size=27`, and
  `observation_dim=90` for the standard base/command/joint/previous-action
  vector shape.
- The training profile keeps the H200 evidence from the guarded run:
  `CUDA_VISIBLE_DEVICES=1`, `physical_gpu=1`,
  `logical_cuda_device=cuda:0`, `n_envs=1024`,
  `env_policy_steps_per_sec=45827.527990`,
  `env_sim_steps_per_sec=183310.111961`,
  `tensor_device_ok=true`, and selected root+joint reset target-only behavior.
  The component-profile evidence records `scene_steps_per_sec=187.421643`
  and `combined_env_policy_steps_per_sec=29547.329979`.
- Local focused verification for the training profile:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_g1_27dof_nohand_profile.py tests/test_robot_profile_loader.py -q -p no:cacheprovider`
  -> 27 passed. The existing 29DoF SONIC loader remains strict and rejects the
  27DoF training profile through the original `RobotProfileError` path.
- Non-blocking warnings: Genesis emitted `torch<2.8.0` warning under
  `torch==2.5.1`; `trimesh` emitted `invalid value encountered in divide`;
  Genesis warned that `left_ankle_roll_link` and `right_ankle_roll_link` have
  dubious masses relative to geometry estimates. These warnings did not block
  build, tensor I/O, selected reset, or stepping.
- GPU memory remained healthy in saved snapshots. The optional `n_envs=4096`
  G1 run ended with physical GPU 1 at 5407 MiB used out of 143771 MiB.
- Result: target G1 official MJCF batched/tensor API path passed. Remaining
  concern is performance/asset quality, not API or H200 environment health.

# Review

Status: passed.

Read-only review initially found two blocking evidence gaps: floating-base
selected reset only covered DOFs, and root velocity device was not emitted. The
probe was fixed and G1 was rerun. Re-review found no blocking or important
findings. G1 evidence now satisfies the subtask requirements, including
selected root qpos plus joint reset, root velocity tensor device evidence,
hard CUDA tensor-device checks, default-before-optimization ordering, and
optional `4096` capacity evidence after `1024` passed.
