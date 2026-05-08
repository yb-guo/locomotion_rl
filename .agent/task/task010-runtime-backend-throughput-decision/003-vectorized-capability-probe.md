# Route

Task: task010-runtime-backend-throughput-decision

Goal: determine whether current Genesis + G1 29DoF path supports the tensor and
batch capabilities needed for large-scale RL.

Scope:

- Probe `n_envs > 1` scene build where practical.
- Probe batched motor target writes where practical.
- Probe batched root/dof state reads where practical.
- Probe selected-env reset support or record the missing API.
- Record whether any probe uses per-env Python loops or per-step CPU readback.

Environment:

- interactive remote sessions must enter `/root/agent_workspace/safe_agent/agent_shell.sh`
- non-interactive remote commands must run through `/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'`
- code, generated files, and intermediate outputs must stay under `/root/agent_workspace/project`
- datasets must be accessed only through `/root/agent_workspace/datasets`
- dataset links must be created with `/root/agent_workspace/safe_agent/link_dataset.sh`
- do not write to or delete from `/mnt/workspace` or `/mnt/workspace1`
- do not manually clear `LD_PRELOAD`

Verify:

- H200 guarded probe reports capability flags or exact failure messages
- blocker, if present, identifies API, asset, reset, tensor IO, or Python loop

No Hack:

- Do not fake vectorized support by looping scalar envs and reporting it as
  batched training support.
- Do not download replacement assets.

Hardware: H200/Linux target via `myserver`.

# Log

- 2026-05-08 local: The throughput probe reports vectorized capability flags
  without wrapping scalar envs in a fake per-env Python loop.
- Local CLI capability check for `n_envs=16`:
  `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.genesis_g1_throughput_probe --asset does-not-exist.xml --n-envs 16 --warmup-policy-steps 1 --measure-policy-steps 1`
  returned parseable output with:
  `status=capability_failure`,
  `capability_failure=GenesisG1SceneBackend currently supports n_envs=1 only`,
  `batched_build_supported=false`,
  `batched_action_write_supported=false`,
  `batched_state_read_supported=false`,
  `selected_reset_supported=false`,
  `cpu_readback_per_step=true`,
  `per_env_python_loop=false`.
- Focused verification:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_genesis_g1_throughput_probe.py tests/test_scalar_g1_runtime.py`
  -> 10 passed, 1 pytest cache warning from Windows `.pytest_cache` access.
- Full local verification:
  `$env:PYTHONPATH='src'; python -m pytest`
  -> 119 passed, 1 pytest cache warning from Windows `.pytest_cache` access.
- 2026-05-08 H200 guarded `n_envs=16` capability probe returned the same
  explicit blocker before scene build:

```text
probe=genesis_g1_physics_throughput
status=capability_failure
backend=cuda
n_envs=16
action_pattern=zero
render_enabled=false
sonic_enabled=false
onnx_enabled=false
planner_enabled=false
gif_enabled=false
capability_failure=GenesisG1SceneBackend currently supports n_envs=1 only
gpu_backend=true
batched_build_supported=false
batched_action_write_supported=false
batched_state_read_supported=false
selected_reset_supported=false
cpu_readback_per_step=true
per_env_python_loop=false
```

- Stop rule reached: single-env throughput was measured, and `n_envs > 1` is
  clearly blocked by the current `GenesisG1SceneBackend` implementation.

# Review

Status: H200 capability evidence collected; read-only review pending.
