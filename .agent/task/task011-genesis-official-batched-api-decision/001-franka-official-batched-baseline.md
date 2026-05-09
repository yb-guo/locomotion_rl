# Route

Task: task011-genesis-official-batched-api-decision

Goal: validate the H200 + Genesis official MJCF batched/tensor API baseline
using the Franka asset before testing locomotion assets.

Scope:

- Use the prepared official Franka asset.
- Build with `n_envs=1,16,256,1024` subject to stop rules.
- Use default asset/morph parameters first.
- Verify tensor action write, tensor state read, and selected joint reset.
- Record build and steady-state metrics separately.
- Record GPU/memory snapshots before and after benchmark sections.

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

- `status=ok|blocked|failed` for each attempted `n_envs`.
- `cuda_visible_devices=1`, `physical_gpu=1`, and `logical_cuda_device=cuda:0`
  are recorded or a GPU isolation blocker is recorded.
- `action_device`, `qpos_device`, and DOF state devices are recorded.
- `tensor_device_ok=true` is required for pass.
- Selected joint reset changes only target envs.
- Throughput metrics use the task-level definitions.

No Hack:

- Do not use render/GIF/SONIC/ONNX/planner.
- Do not enable `convexify`, `decimate`, or `performance_mode` before the
  default baseline.
- Do not continue to Go2 if Franka fails by stop rule.

Hardware: H200/Linux target via `myserver`.

# Log

pending

# Review

Status: pending.
