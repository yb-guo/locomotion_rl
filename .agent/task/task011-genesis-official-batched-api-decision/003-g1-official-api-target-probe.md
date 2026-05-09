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

pending

# Review

Status: pending.
