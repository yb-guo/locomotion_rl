# Route

Task: task011-genesis-official-batched-api-decision

Goal: validate official Genesis locomotion-style URDF and floating-base tensor
behavior using Go2 after Franka passes.

Scope:

- Use the prepared official Go2 asset.
- Build with `n_envs=1,16,256,1024` subject to stop rules.
- Use default asset/morph parameters first.
- Verify tensor action write and tensor root/DOF state reads.
- Verify selected root + joint reset on target envs only.
- Record GPU/memory snapshots and throughput metrics.

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

- Franka subtask passed or this subtask is skipped by stop rule.
- `status=ok|blocked|failed` for each attempted `n_envs`.
- `cuda_visible_devices=1`, `physical_gpu=1`, and `logical_cuda_device=cuda:0`
  are recorded or a GPU isolation blocker is recorded.
- root pose, root velocity, DOF pos, and DOF vel devices are recorded.
- `tensor_device_ok=true` is required for pass.
- Selected root + joint reset changes only target envs.

No Hack:

- Do not use render/GIF/SONIC/ONNX/planner.
- Do not fake reset by resetting all envs.
- Do not continue to G1 if Go2 fails by stop rule.

Hardware: H200/Linux target via `myserver`.

# Log

- 2026-05-09 H200 guarded Go2 runs started only after Franka passed through
  `n_envs=1024`, satisfying the upstream stop rule.
- Read-only review found that the first probe version recorded selected DOF
  reset but not selected floating-base root reset, and did not emit root
  velocity device evidence. The probe was fixed to read `get_vel()` and to
  exercise selected root qpos reset plus selected joint reset. The Go2 matrix
  below was rerun after that fix.
- H200 guarded benchmark command shape:
  `/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/h200-locomotion-lab-task011-genesis-official-batched-api-decision && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src timeout ... python -m h200_locomotion_lab.tools.genesis_official_batched_api_probe --asset-kind go2 --asset /root/agent_workspace/project/genesis_assets/genesis-world-0.4.6/genesis/assets/urdf/go2/urdf/go2.urdf --n-envs ... --warmup-policy-steps 20 --measure-policy-steps 100 --decimation 4'`
- Default Go2 output files:
  `outputs/task011/go2/default_n1.txt`,
  `outputs/task011/go2/default_n16.txt`,
  `outputs/task011/go2/default_n256.txt`,
  `outputs/task011/go2/default_n1024.txt`.

| n_envs | status | build_time_s | measure_time_s | policy_steps_per_sec | env_policy_steps_per_sec | env_sim_steps_per_sec | tensor_device_ok | selected_reset_changes_only_target_envs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | ok | 15.657439 | 0.619804 | 161.341359 | 161.341359 | 645.365435 | true | false (not applicable for one env) |
| 16 | ok | 15.653096 | 0.464884 | 215.107232 | 3441.715713 | 13766.862852 | true | true |
| 256 | ok | 15.509148 | 0.576046 | 173.597121 | 44440.862851 | 177763.451404 | true | true |
| 1024 | ok | 17.131138 | 0.618651 | 161.642149 | 165521.561041 | 662086.244165 | true | true |

- Every default Go2 run emitted:
  `cuda_visible_devices=1`, `physical_gpu=1`,
  `logical_cuda_device=cuda:0`, `backend=cuda`,
  `includes_build_time=false`, `includes_state_read=true`,
  `includes_action_write=true`, `includes_reward=false`,
  `includes_render=false`, and all recorded root/DOF tensor devices as
  `cuda:0`, including `root_vel_device=cuda:0`.
- Selected root qpos plus joint reset was actually exercised for `n_envs>=16`
  and changed only the target env. The `n_envs=1` run records
  `selected_reset_time_s=not_applicable_n_envs_lt_2`.
- Non-blocking warnings: Genesis emitted `torch<2.8.0` warning under
  `torch==2.5.1` and a neutral qpos0 joint-limit warning for the URDF.
- Result: Go2 official URDF/floating-base batched/tensor API baseline passed
  through `n_envs=1024`; stop rule permits G1.

# Review

Status: passed.

Read-only review initially found two blocking evidence gaps: floating-base
selected reset only covered DOFs, and root velocity device was not emitted. The
probe was fixed and Go2 was rerun. Re-review found no blocking or important
findings. Go2 evidence now satisfies the subtask requirements, including
selected root qpos plus joint reset, root velocity tensor device evidence, and
hard CUDA tensor-device checks.
