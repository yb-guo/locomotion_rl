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

pending

# Review

Status: pending.
