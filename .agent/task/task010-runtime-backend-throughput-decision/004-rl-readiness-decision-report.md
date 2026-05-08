# Route

Task: task010-runtime-backend-throughput-decision

Goal: write the decision report that chooses the next RL backend route.

Scope:

- Summarize scalar runtime correctness evidence.
- Summarize H200 Genesis throughput evidence.
- Summarize vectorized/tensor capability evidence.
- Decide whether task011 should be Genesis vectorized PPO, Genesis adaptation
  work, single-env PPO smoke, or a backend switch.
- Record residual risks and exact blockers.

Environment:

- local documentation update
- remote evidence must be copied or summarized from guarded H200 runs

Verify:

- report includes the required metrics and capability flags
- task top-level `Result` remains pending unless all subtasks pass

No Hack:

- Do not claim RL readiness without throughput and vectorized capability
  evidence.
- Do not hide failed probes as pending.

Hardware: local docs plus H200 evidence from prior subtasks.

# Log

## Decision

Status: current repo Genesis G1 backend is not ready for large-scale PPO/RL
training.

Decision route for task011: do not start PPO on the current
`GenesisG1SceneBackend`. Either:

- create a dedicated `VectorizedGenesisBackend` task based on Genesis tensor and
  batched robot-control APIs; or
- switch the next PPO smoke to another backend if immediate training is higher
  priority than Genesis integration.

## Evidence

- Scalar runtime correctness: local focused tests passed (`10 passed`) and full
  local suite passed (`119 passed`).
- H200 focused task010 tests passed (`11 passed in 0.13s`).
- H200 physics-only single-env 5 warmup / 20 measure run:
  - `backend=cuda`, `gpu_backend=true`
  - `n_envs=1`
  - `action_pattern=zero`
  - `render_enabled=false`
  - `sonic_enabled=false`
  - `onnx_enabled=false`
  - `planner_enabled=false`
  - `gif_enabled=false`
  - `build_time_s=92.2665966`
  - `warmup_time_s=5.16665861`
  - `measure_time_s=54.6162031`
  - `warmup_policy_steps=5`
  - `measure_policy_steps=20`
  - `decimation=4`
  - `policy_steps_per_sec=0.366191695`
  - `sim_steps_per_sec=1.46476678`
  - `env_steps_per_sec=0.366191695`
- H200 50 warmup / 500 measure protocol exceeded 900s before producing output.
- H200 `n_envs=16` capability probe returned:
  `GenesisG1SceneBackend currently supports n_envs=1 only`.
- Capability flags from the current backend:
  - `batched_build_supported=false`
  - `batched_action_write_supported=false`
  - `batched_state_read_supported=false`
  - `selected_reset_supported=false`
  - `cpu_readback_per_step=true`
  - `per_env_python_loop=false`

## Interpretation

The current Genesis path is useful for validation, visual evidence, and
single-robot smoke tests, but it should not be used as the PPO/vectorized
training backend. The limiting issues are not SONIC, ONNX, planner subprocess,
rendering, GIF generation, or per-step SONIC observation/history construction;
the timed warmup/measure loop writes motor targets and advances `scene.step()`
directly. The `GenesisG1SceneBackend.reset()` call still performs existing
wrapper observation/history bookkeeping before timing starts. Even with the
timed path narrowed this way, the backend runs far below the rate needed for RL
and exposes no batched state/action/reset path.

Genesis itself is not ruled out. The current repo wrapper is a single-env smoke
backend, and the next Genesis-specific task should inspect and implement the
proper vectorized/tensor APIs instead of extending scalar runtime or looping
scalar envs.

# Review

Status: passed.

- 2026-05-08: First read-only review blocked because the throughput probe used
  `GenesisG1SceneBackend.step()`, which included SONIC history/observation
  bookkeeping in the measured path while reporting `sonic_enabled=false`.
- 2026-05-08: Router fixed the probe to measure direct motor target writes plus
  `scene.step()` and reran local/H200 evidence.
- 2026-05-08: Second read-only review found no blocking issues. Suggestion to
  clarify reset-time wrapper bookkeeping was applied.
