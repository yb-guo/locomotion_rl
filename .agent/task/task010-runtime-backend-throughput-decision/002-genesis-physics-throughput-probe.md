# Route

Task: task010-runtime-backend-throughput-decision

Goal: measure Genesis G1 physics-only stepping speed on H200 without SONIC,
rendering, ONNX, planner subprocess, or GIF/video work.

Scope:

- Add a throughput probe tool with the task benchmark protocol.
- Measure build, warmup, and steady-state times separately.
- Use existing 29DoF G1 asset and existing Genesis backend.
- Start with `n_envs=1`.
- Record action pattern, backend, dt, decimation, steps, and throughput.

Environment:

- interactive remote sessions must enter `/root/agent_workspace/safe_agent/agent_shell.sh`
- non-interactive remote commands must run through `/root/agent_workspace/safe_agent/run_guarded.sh bash -lc 'cd /root/agent_workspace/project/<project> && <command>'`
- code, generated files, and intermediate outputs must stay under `/root/agent_workspace/project`
- datasets must be accessed only through `/root/agent_workspace/datasets`
- dataset links must be created with `/root/agent_workspace/safe_agent/link_dataset.sh`
- do not write to or delete from `/mnt/workspace` or `/mnt/workspace1`
- do not manually clear `LD_PRELOAD`

Verify:

- local parser/unit tests pass
- H200 guarded single-env throughput command records reproducible metrics
- probe output clearly states that render/SONIC/ONNX/planner are disabled

No Hack:

- Do not include render/GIF/video in throughput numbers.
- Do not include ONNX/planner subprocess in throughput numbers.
- Do not change physics parameters only to improve speed.

Hardware: H200/Linux target via `myserver`.

# Log

- 2026-05-08 local: Added
  `h200_locomotion_lab.tools.genesis_g1_throughput_probe` for physics-only
  Genesis G1 throughput probing. The tool prints parseable `key=value`
  metrics and explicitly reports `render_enabled=false`,
  `sonic_enabled=false`, `onnx_enabled=false`, `planner_enabled=false`, and
  `gif_enabled=false`.
- Local tests cover argument/config validation, deterministic action pattern
  generation, metric/rate calculation, lowercase parseable values, and the
  no-Genesis-import capability failure path.
- Focused verification:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_genesis_g1_throughput_probe.py tests/test_scalar_g1_runtime.py`
  -> 10 passed, 1 pytest cache warning from Windows `.pytest_cache` access.
- Full local verification:
  `$env:PYTHONPATH='src'; python -m pytest`
  -> 119 passed, 1 pytest cache warning from Windows `.pytest_cache` access.
- Local CLI capability check:
  `$env:PYTHONPATH='src'; python -m h200_locomotion_lab.tools.genesis_g1_throughput_probe --asset does-not-exist.xml --n-envs 16 --warmup-policy-steps 1 --measure-policy-steps 1`
  -> `status=capability_failure`,
  `capability_failure=GenesisG1SceneBackend currently supports n_envs=1 only`,
  `batched_build_supported=false`.
- Router follow-up fixed `gpu_backend` to emit a boolean capability flag rather
  than echoing the backend name and changed the measured loop to write motor
  targets plus `scene.step()` directly instead of calling
  `GenesisG1SceneBackend.step()`. This keeps SONIC history/observation work out
  of the physics-only measurement path. Focused verification after the fix:
  `$env:PYTHONPATH='src'; python -m pytest tests/test_genesis_g1_throughput_probe.py tests/test_scalar_g1_runtime.py -q -p no:cacheprovider`
  -> 11 passed. Full local verification:
  `$env:PYTHONPATH='src'; python -m pytest -q -p no:cacheprovider`
  -> 120 passed.
- 2026-05-08 H200 guarded focused tests passed in the extracted task010 tree
  after the physics-only-loop fix:
  `PYTHONPATH=src python -m pytest tests/test_genesis_g1_throughput_probe.py tests/test_scalar_g1_runtime.py -q -p no:cacheprovider`
  -> 11 passed in 0.13s.
- 2026-05-08 H200 guarded protocol run with 50 warmup / 500 measure policy
  steps exceeded the local 900s command timeout and left the probe running.
  Router terminated the orphaned probe process before continuing. This is
  treated as evidence that the current single-env path is too slow for the full
  500-step protocol.
- 2026-05-08 H200 guarded 1 warmup / 1 measure diagnostic run passed:

```text
probe=genesis_g1_physics_throughput
status=ok
backend=cuda
gpu_backend=true
n_envs=1
action_pattern=zero
render_enabled=false
sonic_enabled=false
onnx_enabled=false
planner_enabled=false
gif_enabled=false
build_time_s=91.7190992
warmup_time_s=1.63313869
measure_time_s=1.18102473
policy_steps_per_sec=0.846722324
sim_steps_per_sec=3.38688929
env_steps_per_sec=0.846722324
capability_failure=none
batched_build_supported=false
batched_action_write_supported=false
batched_state_read_supported=false
selected_reset_supported=false
cpu_readback_per_step=true
per_env_python_loop=false
```

- 2026-05-08 H200 guarded 5 warmup / 20 measure short steady-state run passed
  after the physics-only-loop fix:

```text
probe=genesis_g1_physics_throughput
status=ok
backend=cuda
gpu_backend=true
n_envs=1
action_pattern=zero
render_enabled=false
sonic_enabled=false
onnx_enabled=false
planner_enabled=false
gif_enabled=false
build_time_s=92.2665966
warmup_time_s=5.16665861
measure_time_s=54.6162031
warmup_policy_steps=5
measure_policy_steps=20
decimation=4
policy_steps_per_sec=0.366191695
sim_steps_per_sec=1.46476678
env_steps_per_sec=0.366191695
capability_failure=none
batched_build_supported=false
batched_action_write_supported=false
batched_state_read_supported=false
selected_reset_supported=false
cpu_readback_per_step=true
per_env_python_loop=false
```

- Genesis emitted high-face-count and SDF preprocessing warnings for the
  current G1 mesh asset, plus the existing `torch<2.8.0` warning. No
  physics-parameter shortcuts, decimation, convexification, render, ONNX, or
  planner work were included in the throughput numbers.

# Review

Status: H200 throughput evidence collected; read-only review pending.
