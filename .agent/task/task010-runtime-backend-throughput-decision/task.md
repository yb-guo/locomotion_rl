# Task 010: Runtime Backend Throughput Decision

## Goal

Decide whether Genesis on the H200 target is a viable large-scale RL training
backend for the current Unitree G1 29DoF route.

This task prioritizes a decision report over broad refactoring. Code changes
must be only large enough to support reproducible evidence.

## Scope

- Add a minimal scalar runtime loop for single-robot SONIC/G1 backend checks.
- Define an `ActionProvider` boundary with fake/sequence providers.
- Prove the scalar runtime can run a log replay backend locally.
- Add a Genesis physics-only throughput probe that excludes SONIC, ONNX,
  planner subprocess, render, and GIF/video work.
- Run H200 Genesis throughput measurements for `n_envs=1` and probe
  `n_envs > 1` support where practical.
- Record whether Genesis exposes training-usable GPU/tensor state/action/reset
  paths for G1.
- Produce an RL readiness decision report that selects the next route.

## Non-Goals

- No PPO training.
- No LocoFormer training or multi-embodiment randomization.
- No full SONIC planner/encoder/decoder provider extraction.
- No real Unitree SDK integration.
- No migration of all historical `tools/` scripts.
- No new robot assets, checkpoints, datasets, upstream repos, or simulator
  version upgrades unless explicitly requested.
- No physics-parameter shortcuts that make throughput numbers unrealistic.

## Performance Contract

Scalar runtime:

- May use dataclasses and Python objects.
- May use clear logging, finite checks, watchdogs, hooks, and safety gates.
- Is for single robot, real robot deployment, smoke tests, and visual/debug
  rollouts.
- Must not become the PPO or vectorized training inner loop.

Training hot path:

- Must use compiled profiles, not per-step YAML reads.
- Must use precompiled index arrays/tensors, not per-step joint-name lookup.
- Must avoid per-step dataclass allocation in the vectorized inner loop.
- Must avoid per-step large GPU tensor readback to CPU.
- Must avoid render, GIF/video, ONNX, planner subprocess, and per-frame prints
  in throughput benchmarks.

Throughput metrics must separate build time, warmup time, and steady-state
measurement time.

## Benchmark Protocol

Default H200 protocol:

- `backend=cuda`
- existing SONIC-compatible 29DoF G1 MJCF asset
- `render=false`
- `sonic=false`
- `onnx=false`
- `planner=false`
- warmup: 50 policy steps
- measure: 500 policy steps
- `sim_dt=0.005`
- `decimation=4`
- `policy_rate_hz=50`
- action patterns: zero, random, sine where practical
- `n_envs=1`, then capability probe for `16`, `64`, `256` if supported

Required metrics:

- `build_time_s`
- `warmup_time_s`
- `measure_time_s`
- `policy_steps_per_sec`
- `sim_steps_per_sec`
- `env_steps_per_sec`
- `gpu_backend`
- `batched_build_supported`
- `batched_action_write_supported`
- `batched_state_read_supported`
- `selected_reset_supported`
- `cpu_readback_per_step`
- `per_env_python_loop`

## Stop Rules

Stop exploration and write the report once any of these is true:

- Single-env physics-only throughput is measured and `n_envs > 1` is clearly
  supported or clearly blocked.
- A concrete Genesis G1 asset/API blocker is found.
- State read or action write requires per-step CPU readback with no obvious
  tensor path.
- H200 Genesis guarded runs fail twice with the same infrastructure or simulator
  error.
- Single-env steady-state throughput is too low for PPO and the cause is not
  render, ONNX, planner subprocess, or logging.

## Subtasks

- `001-scalar-runtime-min-loop.md`
- `002-genesis-physics-throughput-probe.md`
- `003-vectorized-capability-probe.md`
- `004-rl-readiness-decision-report.md`

## Acceptance

- Each subtask has `Route / Log / Review`.
- Local tests pass for new runtime/probe code.
- H200 evidence is recorded through guarded commands when simulator behavior is
  tested on the target.
- The final report states one of:
  - Genesis is ready for a vectorized PPO task.
  - Genesis needs a dedicated vectorized backend/API adaptation task.
  - Genesis is only suitable for validation/visualization for now.
  - The project should switch RL backend for the next PPO task.
- Blocking findings from read-only review are fixed or the task remains
  pending/blocked.

## Result

Status: passed.

Task010 reached the stop rule and produced the requested decision.

Closed scope:

- Added minimal scalar G1 runtime and fake/zero/sequence action providers.
- Added a Genesis G1 physics-only throughput probe.
- Verified local scalar runtime and probe behavior.
- Ran H200 guarded focused tests.
- Measured H200 single-env Genesis G1 physics-only throughput.
- Probed `n_envs=16` capability and recorded the explicit blocker.
- Wrote the RL readiness decision report.

Verification evidence:

- Local full suite: `120 passed`.
- H200 focused task010 tests: `11 passed in 0.13s`.
- H200 single-env physics-only short run:
  `policy_steps_per_sec=0.366191695`,
  `sim_steps_per_sec=1.46476678`,
  `env_steps_per_sec=0.366191695`,
  `build_time_s=92.2665966`,
  `measure_time_s=54.6162031`.
- H200 `n_envs=16` capability probe:
  `GenesisG1SceneBackend currently supports n_envs=1 only`.
- Capability flags:
  `batched_build_supported=false`,
  `batched_action_write_supported=false`,
  `batched_state_read_supported=false`,
  `selected_reset_supported=false`,
  `cpu_readback_per_step=true`.

Decision:

The current repo `GenesisG1SceneBackend` is not ready for large-scale PPO/RL
training. It remains useful for single-robot validation and visual evidence.
The next Genesis-specific task should inspect and implement proper
vectorized/tensor APIs instead of extending scalar runtime or looping scalar
envs.

## Review

Status: passed.

- 2026-05-08: First read-only review found one blocking issue: the throughput
  probe measured through `GenesisG1SceneBackend.step()`, which included
  observation/history work while reporting `sonic_enabled=false`.
- 2026-05-08: Router fixed the measured loop to write motor targets and call
  `scene.step()` directly, then reran local and H200 evidence.
- 2026-05-08: Second read-only review found no blocking issues. Its suggestion
  to clarify reset-time wrapper bookkeeping was applied in the decision report.
